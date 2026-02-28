# LOL Teamfight Lab — Complete Project Documentation

**Version:** 0.1.0
**Target Venue:** IEEE Conference on Games (CoG) 2026
**Language:** Python 3.8+
**License:** MIT (research and educational purposes)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Pipeline](#2-architecture--data-pipeline)
3. [Directory Structure](#3-directory-structure)
4. [Technologies & Dependencies](#4-technologies--dependencies)
5. [Setup & Installation](#5-setup--installation)
6. [Configuration](#6-configuration)
7. [Fight Detection Algorithm](#7-fight-detection-algorithm)
8. [Feature Engineering](#8-feature-engineering)
9. [Model Architectures](#9-model-architectures)
10. [Ensemble & Fusion Methods](#10-ensemble--fusion-methods)
11. [Domain Knowledge Improvements (7 Treatments)](#11-domain-knowledge-improvements-7-treatments)
12. [Data Splitting & Leakage Prevention](#12-data-splitting--leakage-prevention)
13. [Training & Evaluation](#13-training--evaluation)
14. [Ablation Study Framework](#14-ablation-study-framework)
15. [Usage Guide](#15-usage-guide)
16. [Testing](#16-testing)
17. [Design Principles](#17-design-principles)
18. [Project Statistics](#18-project-statistics)

---

## 1. Project Overview

LOL Teamfight Lab is a comprehensive machine learning pipeline for predicting **League of Legends teamfight outcomes** using match timeline data from the Riot Games API. The project formulates teamfight prediction as a spatio-temporal multivariate time-series classification task over heterogeneous player interaction graphs.

### What It Does

1. **Detects teamfights** from raw game event data via kill-cluster-based temporal clustering with spatial validation
2. **Extracts multi-modal temporal features** — per-player stats (87-dim), team-level globals (27-dim), event sequences (48-dim), and categorical embeddings (champion, runes, summoner spells)
3. **Trains 15+ model architectures** spanning tabular (LightGBM), deep sequential (BiGRU, BiLSTM, Transformer, TCN, Mamba), graph neural networks (GCN, GraphSAGE, GATv2, MPNN, ST-GNN), and fusion architectures
4. **Ensembles predictions** through stacking, out-of-fold meta-learning, factorial search, and a novel Layered Fusion architecture

### Why It Matters

Teamfights are the decisive moments in competitive League of Legends — a MOBA game with 150M+ monthly active players. Predicting fight outcomes *before they unfold* requires understanding the complex interplay of positioning, power spikes, cooldowns, item builds, and team composition across a 16,000 × 16,000 unit map with 10 simultaneously moving players.

Prior approaches operated at the match level (who wins the whole game). This project tackles the harder **within-game prediction** problem: given 60 seconds of game state leading up to a teamfight, can we predict which team wins that specific engagement?

---

## 2. Architecture & Data Pipeline

```
Match JSONs (detail + timeline from Riot API)
        │
        ▼
  [1] data.cache_io.prebuild_cache()
        │   Parse 60s frame snapshots + ms-level events → NumPy arrays
        ▼
  Preprocessed Match Cache (NPZ + JSON metadata)
        │
        ▼
  [2] gameplay.fights.detect_fights()
        │   Kill clustering → spatial validation → fight characterization
        ▼
  core.fight_types.FightRef (unique fight identifiers with timestamps)
        │
        ▼
  [3] data.index_split.build_fight_index() → split_refs()
        │   Match-grouped, patch-stratified train/val/test splits
        ▼
  Train / Val / Test FightRef Lists
        │
        ▼
  [4] data.dataset.InMemoryFightDataset
        │── gameplay.pipeline.build_ms_sequence()       [60s → 12×5s bins]
        │── gameplay.features.build_sequence_features() [feature extraction]
        │── collate_batch()                             [batching + graph pooling]
        ▼
  [5] Model Training
        │── train.baseline: LightGBM (tabular gradient boosting)
        │── train.deep:     RNN / Transformer / TCN / Mamba (sequential)
        │── train.deep:     GNN / GAT / MPNN / ST-GNN (graph)
        ▼
  [6] Ensemble Fusion (train.fusion)
        │── Simple stacking (logistic regression meta-learner)
        │── Out-of-fold stacking (K-fold train meta-learner)
        │── Factorial stacking (all subset combinations)
        │── Greedy forward selection
        │── Layered Fusion (gated global + graph + event streams)
        ▼
  [7] Reports (AUC, AP, Brier, minutewise, situation-aware, bootstrap CI)
```

### Key Design Invariants

- **Ref-key alignment**: Predictions are matched to ground truth by `"match_id|t_start_ts=<ms>"` string keys, never by batch position — safe under shuffling and dropped samples
- **No future leakage**: Node and global features use strictly-before 60s snapshots (piecewise-constant); the label window `[engage_ts, engage_ts + horizon)` is never observed by the model
- **Match-grouped splits**: All fights from one match stay in the same split partition
- **XY zeroed in model input**: Player positions used only for fight detection spatial checks

---

## 3. Directory Structure

```
LOL_teamfight_Lab/
│
├── main.py                         # Entry point wrapper → runner.main()
├── runner.py                       # CLI argument parser & main orchestrator
├── experiment_runner.py            # 6-phase systematic ablation study runner
├── paper_experiment_plan.py        # CoG 2026 deadline experiment plan
│
├── pyproject.toml                  # Build & package configuration (setuptools)
├── requirements.txt                # Pinned dependency list
├── __init__.py                     # Package metadata (v0.1.0)
├── .gitignore                      # Git exclusions
│
├── core/                           # Configuration, contracts & utilities
│   ├── config.py                   # Central CFG dataclass — 150+ hyperparameters (SSoT)
│   ├── config_legacy.py            # Deprecated config (reference only)
│   ├── contract.py                 # Contract validation (feature + time)
│   ├── feature_contract.py         # Feature dimension definitions
│   ├── time_contract.py            # Time/index contract for legacy compatibility
│   ├── fight_types.py              # FightRef dataclass & ref_key()
│   ├── roles.py                    # Lane/role assignment (TOP/JNG/MID/BOT/SUP)
│   ├── diagnostics.py              # Fight detection debug dumps
│   ├── improvements.py             # 7 domain-knowledge enhancements
│   ├── interpolation.py            # Alpha-curve functions (linear/cosine/exponential/cubic)
│   ├── timeutils.py                # Time-based calculations (ctx_ms, bin_ms, horizon_ms)
│   ├── common.py                   # Shared math (safe_float, logit, log1p_norm)
│   ├── common_torch.py             # PyTorch utilities (autocast, nan_to_num, NODE_IDX)
│   └── utils.py                    # General utilities (metrics, seeding, I/O)
│
├── data/                           # Data loading, caching & splitting
│   ├── cache_io.py                 # Match cache I/O (NPZ + JSON build/load)
│   ├── ram_cache.py                # In-RAM LRU cache for loaded match packs
│   ├── file_io.py                  # File utilities (CSV, JSON, directory management)
│   ├── dataset.py                  # InMemoryFightDataset (PyTorch Dataset)
│   ├── index_split.py              # Data splitting strategies (multi_patch, patch_holdout, etc.)
│   ├── indexing.py                 # Match indexing, leakage checks, patch counting
│   ├── events_index.py             # Event timestamp indexing & lookup
│   ├── labels.py                   # Ground-truth label alignment (label maps)
│   └── logits.py                   # Model prediction map management
│
├── gameplay/                       # Fight detection & feature engineering
│   ├── fights.py                   # Fight detection engine (teamfight_v2)
│   ├── fight_clustering.py         # Temporal kill clustering algorithm
│   ├── fight_postmerge.py          # Post-merge validation (spacing, overlap, duration)
│   ├── fight_analysis.py           # Fight outcome, importance, engagement computation
│   ├── fight_metrics.py            # Team gold, resource changes, team assignment
│   ├── pipeline.py                 # Core temporal feature building (build_ms_sequence)
│   ├── pipeline_cache.py           # Cache parsing (NPZ → minute-level arrays)
│   ├── pipeline_interp.py          # Interpolation for node/global features
│   ├── features.py                 # Feature builders, normalizers, role reordering
│   ├── feature_spatial.py          # Spatial feature computation (distances, zones)
│   ├── labels.py                   # Label computation (kill_survival scoring)
│   ├── event_aggregation.py        # Per-bin event counting
│   └── event_tokens.py             # Discrete event token construction (for cross-attention)
│
├── train/                          # Model definitions & training
│   ├── models.py                   # Model factory & 15+ architecture definitions
│   ├── deep.py                     # Deep learning training harness (train loop, AMP, etc.)
│   ├── deep_eval.py                # Evaluation utilities (logit extraction, metrics)
│   ├── baseline.py                 # LightGBM tabular baseline + recency weighting
│   ├── fusion.py                   # Stacking, factorial, greedy forward, refit
│   ├── fusion_calibration.py       # Temperature scaling, ECE computation
│   ├── fusion_helpers.py           # StackingResult, meta-learner fitting
│   ├── graph_encoder.py            # GNN encoder (GCN/GraphSAGE/GAT/MPNN) + pooling
│   ├── node_adapter.py             # Node feature adaptation (embedding + projection)
│   ├── temporal_encoders.py        # RNN/Transformer/TCN/Mamba temporal modules
│   ├── layered_spec.py             # Layered Fusion inline alias parser
│   ├── model_registry.py           # Model registration & alias resolution
│   ├── speed_config.py             # Hardware speed profiles (RTX 50xx, etc.)
│   └── speed.py                    # Performance profiling & torch optimizations
│
├── app/                            # Orchestration & analysis
│   ├── experiment.py               # Main training loop orchestrator (run())
│   ├── experiment_helpers.py       # Model inference, alias resolution, subsampling
│   ├── experiment_exec_helpers.py  # Execution utilities
│   ├── experiment_io.py            # Results I/O (JSON serialize/deserialize)
│   ├── experiment_runner_io.py     # Ablation runner CLI parser
│   ├── experiment_runtime.py       # Single experiment execution
│   ├── experiment_stats.py         # Bootstrap CI, Holm-Bonferroni, safe_mean/std
│   ├── experiment_types.py         # ExperimentResult, AblationSummary, Treatment defs
│   ├── analysis.py                 # Fight analysis utilities
│   ├── analysis_metrics.py         # Metric computation helpers
│   ├── analysis_plotting.py        # Visualization (matplotlib)
│   ├── analysis_reporting.py       # Report generation
│   ├── split_reports.py            # Train/val/test split analysis reports
│   └── detection_quality_report.py # Fight detection quality metrics & comparison
│
├── analysis/                       # Post-hoc analysis & ablation
│   ├── analysis.py                 # General analysis tools
│   ├── feature_ablation.py         # Comprehensive feature contribution analysis (Phase 6)
│   └── teamfight_duration_investigation.py  # Data quality root-cause analysis
│
├── tests/                          # Unit tests (pytest)
│   ├── test_common.py              # Math utilities (sigmoid, logit, log1p)
│   ├── test_config.py              # CFG dataclass defaults & validation
│   ├── test_experiment_runner.py   # Ablation runner (bootstrap CI, seeds)
│   ├── test_fight_types.py         # FightRef construction & key stability
│   ├── test_index_split.py         # Data splitting & leakage checks
│   ├── test_utils.py               # General utility tests
│   ├── test_teamfight_v2.py        # Fight detection algorithm tests
│   ├── test_interpolation.py       # Interpolation curve tests
│   ├── test_postmerge_and_start_offset.py  # Post-merge validation tests
│   ├── test_detection_quality_report.py    # Detection quality metric tests
│   ├── test_feature_ablation.py    # Feature ablation analysis tests
│   ├── test_runner_paper_preset.py # Paper preset CLI tests
│   └── test_speed_profile.py       # Speed profile configuration tests
│
└── docs/                           # Documentation
    ├── PROJECT.md                  # This file — comprehensive project overview
    ├── pipeline.md                 # End-to-end pipeline architecture (with diagrams)
    ├── CoG2026_Paper.md            # IEEE CoG 2026 conference paper draft
    └── investigation_teamfight_duration.md  # Data quality investigation
```

---

## 4. Technologies & Dependencies

### Core Stack

| Category | Tool | Version | Purpose |
|----------|------|---------|---------|
| **Language** | Python | ≥ 3.8 | Primary language |
| **Deep Learning** | PyTorch | ≥ 2.0.0 | Neural network training (RNN, GNN, Transformer) |
| **Gradient Boosting** | LightGBM | ≥ 3.3.0 | Tabular baseline model |
| **Numerical** | NumPy | ≥ 1.21.0 | Array operations, feature engineering |
| **ML Utilities** | scikit-learn | ≥ 1.0.0 | Metrics, scaling, logistic regression meta-learner |

### Optional Dependencies

| Category | Tool | Version | Purpose |
|----------|------|---------|---------|
| **Explainability** | SHAP | ≥ 0.41.0 | Feature importance analysis |
| **Visualization** | matplotlib | ≥ 3.5.0 | Plots and charts |
| **Data Analysis** | pandas | ≥ 1.3.0 | Tabular data manipulation |
| **Statistics** | scipy | ≥ 1.7.0 | Statistical tests, clustering |
| **Testing** | pytest | ≥ 7.0.0 | Unit test framework |
| **Coverage** | pytest-cov | ≥ 4.0.0 | Test coverage reporting |

### Hardware Acceleration

| Feature | Description |
|---------|-------------|
| **CUDA GPU** | GPU-accelerated training (recommended) |
| **Mixed Precision (AMP)** | bf16/fp16 automatic mixed precision |
| **TF32** | TensorFloat-32 on Ampere+ GPUs (RTX 30xx/40xx/50xx) |
| **torch.compile** | PyTorch 2.0 kernel fusion (optional) |
| **Speed Profiles** | Pre-configured hardware presets (`rtx5080`, `aggressive`) |

---

## 5. Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended, not required)
- League of Legends match data in JSON format (detail + timeline files from Riot API)

### Install Dependencies

```bash
# Editable install with all optional dependencies (recommended for development)
pip install -e ".[all]"

# Or minimal install using requirements.txt
pip install -r requirements.txt

# PyTorch with CUDA 11.8 support (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configure Data Paths

Set environment variables or edit `core/config.py`:

```bash
export LOL_DETAIL_DIR="/path/to/match/details"        # Match detail JSON files
export LOL_TIMELINE_DIR="/path/to/match/timelines"     # Match timeline JSON files
export LOL_OUTPUT_ROOT="/path/for/outputs"             # Model outputs, reports, logs
```

### Verify Installation

```bash
# Run the test suite to verify everything works
pytest -v

# Quick sanity check with 10 matches
python main.py --mode all --max_matches 10 --seed 7
```

---

## 6. Configuration

All hyperparameters live in `core/config.py` as a single `CFG` dataclass — the project's **single source of truth**.

### Key Configuration Sections

| Section | Parameters | Description |
|---------|-----------|-------------|
| **Data Paths** | `DETAIL_DIR`, `TIMELINE_DIR`, `OUTPUT_ROOT` | Input/output directories |
| **Cache** | `CACHE_VERSION`, `CACHE_DIR` | Feature schema versioning for cache invalidation |
| **Fight Detection** | `TF2_KILL_CLUSTER_GAP_MS`, `TF2_VALIDITY_RADIUS`, etc. | Teamfight detection thresholds |
| **Features** | `NODE_FEATURE_NAMES`, `GLOBAL_FEATURE_NAMES`, etc. | Feature contract definitions |
| **Model List** | `MODEL_LIST`, model aliases | Active models for training |
| **Training** | `LR`, `EPOCHS`, `PATIENCE`, `BATCH_SIZE`, `DROPOUT` | Training hyperparameters |
| **Splitting** | `SPLIT_MODE`, `VAL_RATIO`, `SEEDS` | Data partitioning strategy |
| **Improvements** | `USE_FOCAL_LOSS`, `USE_GAME_PHASE`, etc. | Domain enhancement toggles |

### Essential Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LR` | 5e-4 | Learning rate |
| `EPOCHS` | 15 | Maximum training epochs |
| `PATIENCE` | 3 | Early stopping patience |
| `BATCH_SIZE` | 32 | Training batch size |
| `DROPOUT` | 0.20 | General dropout rate |
| `RNN_HIDDEN` | 128 | RNN hidden dimension |
| `GNN_DIM` | 96 | GNN hidden dimension |
| `GNN_DROPOUT` | 0.25 | GNN-specific dropout |
| `CONTEXT_MS` | 60,000 | Observation window (60 seconds) |
| `BIN_MS` | 5,000 | Time bin size (5 seconds) |
| `HORIZON_MS` | 60,000 | Label window duration |
| `SEEDS` | [7, 42, 123, 256, 512] | Bootstrap seeds for CI |
| `VAL_RATIO` | 0.15 | Validation split fraction |

### CLI Overrides

Most configuration can be overridden via command-line arguments:

```bash
python runner.py --seed 42 --batch_size 64 --models rnn_bigru,gnn_graphsage \
    --split_mode patch_holdout --amp --tf32 --speed_profile rtx5080
```

---

## 7. Fight Detection Algorithm

**Entry point:** `gameplay/fights.py::detect_fights()` → `detect_fights_teamfight_v2()`

The fight detection pipeline transforms raw match events into validated teamfight instances. It operates in 6 steps:

### Step 1: Build 5-Second Position Grid

Riot API provides player positions at 60-second intervals. The grid interpolates positions to 5-second resolution using configurable alpha curves (default: exponential with k=3).

```
Riot API: 60s frames                  Dense 5s grid
  ┌─ 0:00 ─┬─ 1:00 ─┬─ 2:00 ─┐        ┌─ 0:00 ─ 0:05 ─ 0:10 ─ ... ─ 0:55 ─ 1:00 ─
  │ (x,y)  │ (x,y)  │ (x,y)  │  ───►   │  interpolated XY at every 5-second mark
  └────────┴────────┴────────┘        └─ for all 10 players
```

A pre-kill override layer adjusts participant positions to interpolate toward kill locations, improving spatial accuracy for engagement detection.

### Step 2: Cluster Kills by Temporal Proximity

```
gap_ms = 18,000 (18 seconds)

Timeline:  K1(120s)  K2(125s)  K3(131s)       K4(180s)  K5(185s)  K6(192s)
           │←──5s──→│←──6s──→│                 │←──5s──→│←──7s──→│
           │    within 18s gap                  │   within 18s gap
           └──── Cluster A ─────┘              └──── Cluster B ─────┘
                         ▲ 49s gap ▲
                        (> 18s → split)
```

### Step 3: Validate Each Cluster as Teamfight

For each kill cluster:

1. **Compute engage time**: `engage_ts = first_kill_ts − 10,000ms`
2. **Check context bounds**: Must be ≥60s into the game; horizon cannot exceed game end
3. **Check alive count**: Both teams must have ≥2 alive champions at engage time
4. **Spatial validation**: At `engage_ts`, count players within radius=1,800 of fight center; require ≥2 blue AND ≥2 red

```
  Valid teamfight:                    Rejected (pick):
  ┌────── r=1800 ──────┐             ●B1        ⊕ center
  │ ●B1  ●B2           │                        ●R1
  │       ⊕ center     │             blue_in=1 < 2 ✗
  │ ●R1  ●R2           │
  └─────────────────────┘
  blue=2, red=2 ✓
```

### Step 4: Collect Interactions

Non-kill events within radius=3,000 of fight center during `[engage_ts, last_kill_ts]` are counted as fight interactions. Objective and structure events are excluded (tracked separately in Step 5).

### Step 5: Post-Fight Outcome (45-Second Window)

After the last kill, a 45-second window captures consequences: objectives taken, towers destroyed, gold differential changes.

### Step 6: Classify & Score

Each fight is classified by spatial location:

| Classification | Condition |
|---------------|-----------|
| `objective_baron` | Center < 1,500 units from Baron pit |
| `objective_dragon` | Center < 1,500 units from Dragon pit |
| `objective_riftherald` | Center < 1,500 units from Rift Herald |
| `tower_dive` | Center < 1,000 units from a tower |
| `base_fight` | Center < 3,000 units from a base |
| `teamfight` | ≥ 8 proximity pairs |
| `skirmish` | ≥ 4 proximity pairs |
| `pick` | Otherwise |

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TF2_KILL_CLUSTER_GAP_MS` | 18,000 | Max gap between kills in same cluster |
| `TF2_ENGAGE_PRE_KILL_MS` | 10,000 | Engagement offset before first kill |
| `TF2_VALIDITY_RADIUS` | 1,800 | Teamfight validation radius (game units) |
| `TF2_INTERACTION_RADIUS` | 3,000 | Interaction counting radius |
| `TF2_POST_FIGHT_WINDOW_MS` | 45,000 | Post-fight outcome window |
| `TF2_MIN_PER_TEAM` | 2 | Minimum champions per team in radius |
| `FIGHT_MIN_GAP_MS` | 60,000 | Minimum spacing between fights |
| `MAX_MERGED_FIGHT_DURATION_MS` | 120,000 | Maximum fight duration cap |

---

## 8. Feature Engineering

### Observation Window

Each training sample represents a 60-second observation window before a teamfight, discretized into L=12 bins of 5 seconds each:

```
  ◄────────── observation window (60s) ──────────►
  │ bin0  bin1  bin2  ...  bin10  bin11            │
  │[0-5s][5-10s][10-15s]      [50-55s][55-60s]    │
  start_ms                                     engage_ts
  (engage_ts − 60s)                                │
                                    ◄── label window (60s) ──►
                                    engage_ts          label_end_ts
```

### Feature Tensors

At each of the 12 time bins, four feature tensors are constructed:

#### Node Features: `node_seq ∈ ℝ^{L × N × F_node}`

87-dimensional per-player features (N=10 players, L=12 time steps):

| Feature Group | Dim | Description |
|--------------|-----|-------------|
| Position | 2 | Normalized (x, y) — **zeroed in model input** |
| Resources | 5 | level, XP, currentGold, totalGold, goldPerSecond |
| CS | 2 | lane minions killed, jungle minions killed |
| Health | 3 | HP%, MP%, alive (0/1) |
| Identity | 2+ | champion_id, champion_name_id (categorical → embedding) |
| Summoner Spells | 2 | summoner_spell_1_id, summoner_spell_2_id |
| Runes | 8 | primary rune 1-4, sub rune 1-2, style IDs |
| Buffs | 4 | has_baron, has_elder, baron_remain_norm, elder_remain_norm |
| Cooldowns | 1 | ult_level_norm |
| Dragon Soul | 6 | soul_infernal, ocean, mountain, cloud, hextech, chemtech |
| Champion Stats | 25 | armor, AD, AP, MR, attack_speed, crit, etc. |
| Damage Stats | 12 | total/physical/magic damage done/taken to champions |

#### Global Features: `glob_seq ∈ ℝ^{L × F_global}`

27-dimensional team-level features:

| Feature | Description |
|---------|-------------|
| `time_norm` | Game time normalized to [0, 1] |
| `blue_ban_0..4`, `red_ban_0..4` | Champion ban IDs |
| `goldDiff`, `xpDiff` | Gold and XP differentials |
| `avgLevelDiff`, `csDiff_total` | Level and CS differentials |
| `aliveDiff` | Alive champion differential |
| `killDiff_cum`, `towerDiff_cum` | Cumulative kill/tower differentials |
| `dragonDiff_cum`, `baronDiff_cum` | Cumulative objective differentials |
| `inhibDiff_cum`, `heraldDiff_cum` | Cumulative structure/objective diffs |
| `plateDiff_cum`, `hordeDiff_cum` | Plate and void grub differentials |

#### Event Features: `ev_seq ∈ ℝ^{L × F_event}`

48-dimensional per-bin event counts:

| Category | Features |
|----------|----------|
| Combat | kills, bounties, shutdowns, killstreaks, multikills, aces (per team) |
| Objectives | dragon, baron, herald, atakhan, horde takes (per team) |
| Structures | tower kills, inhibitor kills, turret plates (per team) |
| Vision | wards placed/killed, control wards placed/killed (per team) |
| Items | item purchases, sells, undos (per team) |

#### Event Tokens (for Cross-Attention Models)

Up to 64 discrete event tokens, each with:
- `event_type` (int64): event category hash
- `event_actor` (int64): acting participant ID
- `event_team` (int64): team (0=blue, 1=red, 2=unknown)
- `event_cont` (float32, dim=12): [t_rel, dt_end, x, y, val, ...importance features]
- `event_mask` (float32): 1=real, 0=padding

### Label Computation

**Primary label** (binary classification):

```
Score = W_KILL × (blue_kills − red_kills) + W_ALIVE × (blue_alive − red_alive)

W_KILL  = 1.0   (kill differential weight)
W_ALIVE = 0.3   (alive-at-end differential weight)

y = 1 if Score > 0  (blue wins)
y = 0 if Score < 0  (red wins)
Ties → dropped or random assignment (seeded)
```

**Auxiliary regression targets** (for multi-task learning):

| Target | Normalization | Description |
|--------|--------------|-------------|
| `y_kill_diff` | kill_diff / 5.0 | Normalized kill differential |
| `y_gold_diff` | gold_diff / 1000.0 | Normalized gold swing |
| `y_obj_diff` | obj_diff / 5.0 | Normalized objective differential |

### Feature Normalization

| Feature Type | Normalization | Method |
|-------------|---------------|--------|
| Positions (x, y) | MAP_MAX (16,000) | Division + zeroing in model input |
| Level | 18.0 | Division |
| Gold/XP/CS | Deterministic denominator | Piecewise-constant clip to [-10, 10] |
| Champion stats (cs_*) | Per-stat denominator | Linear clip |
| Damage stats (ds_*) | Per-stat denominator | Linear clip |
| HP%, MP% | Already [0, 1] | Clip |

### Feature Sets

The pipeline supports multiple feature set configurations:

| Feature Set | Composition |
|------------|-------------|
| `global_only` | Global features + spatial |
| `global_events` | Global + events + items + spatial |
| `node_personal` | Flattened node features + minimal events + spatial |
| `full` | All node + global + events + items + spatial |
| `tri_modal` | Node (graph) + macro (sequence) + tabular (aggregated) |

---

## 9. Model Architectures

### Baseline

**LightGBM** — Gradient-boosted decision trees on tabular features
- 5,000 estimators, learning rate 0.03, max depth 6
- Features: aggregated per-player stats, team differentials, event counts
- Recency weighting: `w_i = exp((patch_i − patch_min) / τ)` to handle patch covariate shift

### Deep Sequential Models

| Model | Key Architecture | Description |
|-------|-----------------|-------------|
| **BiGRU** (`rnn_bigru`) | Bidirectional GRU | Captures forward and backward temporal dynamics |
| **BiLSTM** (`rnn_bilstm`) | Bidirectional LSTM | Long short-term memory with gating |
| **Transformer** (`rnn_transformer`) | Multi-head self-attention | Global temporal dependencies via attention |
| **TCN** (`rnn_tcn`) | Temporal convolutional network | Dilated causal convolutions over time |
| **Mamba** (`rnn_mamba`) | State-space model | Efficient long-range sequence modeling |

All sequential models process `extra_seq ∈ ℝ^{L × D}` (macro + spatial features) through their temporal encoder, followed by temporal attention pooling across the 12 time steps and a classification head.

### Hybrid h₀ Models

**Hybrid BiGRU / BiLSTM** — Project tabular features into RNN initial hidden state:
```
h₀ = Linear(tab_features)  →  RNN(extra_seq, h₀)  →  classifier
```

### Graph Neural Networks

| Model | Key Architecture | Description |
|-------|-----------------|-------------|
| **GCN** (`gnn_gcn`) | Graph convolution | Spectral neighborhood aggregation |
| **GraphSAGE** (`gnn_graphsage`) | Sampling + aggregation | Inductive graph learning |
| **GATv2** (`gnn_gatv2`) | Graph attention v2 | Multi-head attention with hard-mask on A==0 edges |
| **MPNN** (`gnn_mpnn`) | Message passing | Edge-attribute-aware message functions |

**Adjacency matrix construction:**
```
A_ij = exp(-d²_ij / 2σ²)      (soft Gaussian from Euclidean distance)
A^role_ij = A_ij × R[role(i), role(j)]   (optional role-aware weighting)
```

GNN models process `node_seq ∈ ℝ^{L × 10 × F_node}` with per-timestep graph operations, then pool across both nodes (team-level) and time (temporal attention).

### Spatio-Temporal Models

| Model | Description |
|-------|-------------|
| **ST-GNN** (`gnn_stgnn`) | Joint spatial (GNN) + temporal (RNN) processing |
| **ST-GCN** (`stgcn`) | GCN with temporal convolution layers |
| **ST-Mamba** (`stgnn_mamba`) | Spatial GNN + Mamba temporal encoder |
| **Event Cross-Attention** (`event_xattn`) | Cross-attention over discrete event tokens |
| **Multi-Scale Dynamic Graph** (`ms_dyngraph`) | Dynamic graph at multiple spatial scales |

### Layered Fusion Architecture

The novel **Layered Fusion** architecture combines three independent streams through gated projection:

```
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │   Global     │     │    Graph     │     │    Event     │
  │ (BiGRU/LSTM/ │     │ (GraphSAGE/  │     │  (Attention/ │
  │  Transformer/│     │  GATv2/MPNN) │     │   XAttn/     │
  │  TCN/Mamba)  │     │              │     │   Mean)      │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                    │                    │
         ▼                    ▼                    ▼
    h_global             h_graph              h_event
         │                    │                    │
         └────────┬───────────┘────────────────────┘
                  │
            Gated Projection
            g = σ(W_g · [h_global; h_graph; h_event])
            h_fused = g ⊙ [h_global; h_graph; h_event]
                  │
            Classification Head
                  │
            logit → P(blue wins)
```

**Inline specification syntax:**
```bash
# BiGRU + GraphSAGE + Attention + LightGBM logit
--models layered_fusion@global=bigru+gnn=graphsage+event=attn+logit=1

# Sweep all combinations
--fusion rnn_all+gnn_all+event_all+logit_all
```

---

## 10. Ensemble & Fusion Methods

### Simple Stacking

- Fit logistic regression meta-learner on TRAIN logits
- Evaluate on VAL/TEST
- Base models: any combination of LightGBM + deep models

### Out-of-Fold (OOF) Stacking

- K-fold training of meta-learner on TRAIN logits
- VAL metric: meta fitted on TRAIN only (unbiased)
- TEST metric: meta fitted on TRAIN+VAL (deployment-like)

### Factorial Stacking

- Enumerate all r-element subsets of candidate models (min_k to max_k)
- Optional anchor constraint (e.g., always include LightGBM)
- Select best subset by VAL AUC

### Greedy Forward Selection

1. Pre-prune highly correlated base models (correlation > 0.9)
2. Pick anchor by best single-model VAL AUC
3. Greedily add models that maximize stacked VAL AUC
4. Stop when no improvement exceeds threshold

### Calibration

- **Temperature scaling**: Post-hoc calibration of prediction confidence
- **ECE (Expected Calibration Error)**: Monitoring for calibration quality
- **Patch-aware calibration**: Separate temperature per game patch

---

## 11. Domain Knowledge Improvements (7 Treatments)

Implemented in `core/improvements.py`, each treatment addresses a specific aspect of the prediction problem. They are evaluated through a systematic ablation study.

### T1: Focal Loss

Down-weights easy examples, focusing learning on margin cases near the decision boundary.

```
ℒ_FL = −α_t (1 − p_t)^γ log(p_t)

Default: γ = 2.0, α = 0.25
```

### T2: Game Phase Encoding

Injects early/mid/late game phase signals as smooth probability features:

```
φ_early(t) = σ((14 − t) / τ)
φ_mid(t)   = σ((t − 10) / τ) · σ((28 − t) / τ)
φ_late(t)  = σ((t − 22) / τ)

τ = 3.0 (transition smoothness)
```

### T3: Temporal Attention Pooling

Learned weighting across the 12 time bins instead of mean/last-step pooling:

```
α_t ∝ exp(w^T tanh(W_a h_t))
h_pool = Σ_t α_t · h_t
```

### T4: Momentum Statistics

Captures short-term vs long-term trends to detect power spikes:

```
δ = μ_short − μ_long    (5s vs 30s windows)
```

### T5: Role-Aware Adjacency

Encodes lane matchup interactions into the GNN adjacency matrix:

```
A^role_ij = A^dist_ij × R[role(i), role(j)]

R: 5×5 matrix for TOP/JNG/MID/BOT/SUP interactions
```

### T6: Multi-Task Learning

Auxiliary regression losses on gold and kill differentials provide richer gradient signal:

```
ℒ_total = ℒ_fight + λ₁ · MSE(pred_gold, y_gold_diff) + λ₂ · MSE(pred_kill, y_kill_diff)
```

### T7: Label Smoothing

Regularization for prediction calibration:

```
y_smooth = y · (1 − ε) + ε / 2

ε = 0.05 (default)
```

---

## 12. Data Splitting & Leakage Prevention

### Split Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `multi_patch` | Stratified by patch, grouped by match_id | Default — balanced representation |
| `group_match` | Grouped by match_id only | When patch info unavailable |
| `patch_forward` | Train on older patches, test on newest | Temporal generalization |
| `patch_holdout` | Specific patches held out for val/test | Paper experiments |
| `random` | Random stratified split | Baseline comparison |

### Default Ratios

- **Train**: 70%
- **Validation**: 15%
- **Test**: 15%

### Leakage Prevention

1. **Match grouping**: All fights from one match stay in the same split partition — prevents cross-split data leakage
2. **Patch stratification**: Each split has proportional representation of game patches — prevents distribution shift artifacts
3. **Leakage check**: `check_split_leakage()` verifies no match_id appears in multiple splits
4. **Test set discipline**: Test set evaluated only once in Phase 5 of the ablation study

### Bootstrap Confidence Intervals

5-seed runs with seeds `[7, 42, 123, 256, 512]` for reporting mean ± std with 95% CI.

---

## 13. Training & Evaluation

### Training Loop

```
For each model in model_list:
    1. Build InMemoryFightDataset for train/val/test splits
    2. Construct DataLoader with collate_batch()
    3. Train with:
       - AdamW optimizer (lr=5e-4, weight_decay=1e-5)
       - Optional AMP mixed precision (bf16/fp16)
       - Early stopping (patience=3, monitor=val_loss)
       - Optional learning rate scheduler
    4. Evaluate on val set, compute logit map
    5. If best val AUC → save checkpoint
    6. Evaluate final checkpoint on test set
```

### Loss Function

```
ℒ = BCE(logit, y)                              # primary classification
  + λ_kill × MSE(pred_kill, y_kill_diff)        # auxiliary kill regression
  + λ_gold × MSE(pred_gold, y_gold_diff)        # auxiliary gold regression
  + λ_obj  × MSE(pred_obj,  y_obj_diff)         # auxiliary objective regression
```

Optional focal loss replaces BCE for hard-example mining.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under ROC Curve (primary metric) |
| **AP** | Average Precision |
| **Accuracy** | Classification accuracy at threshold 0.5 |
| **Precision / Recall / F1** | Per-class and macro |
| **Brier Score** | Calibration quality (lower is better) |
| **ECE** | Expected Calibration Error |

### Subgroup Analysis

- **By game minute**: early (< 15 min) / mid (15-25 min) / late (> 25 min)
- **By gold state**: close (< 2,000) / moderate / stomp (> 5,000)
- **By patch**: per-patch performance tracking
- **By fight type**: teamfight vs skirmish vs objective

---

## 14. Ablation Study Framework

The experiment runner (`experiment_runner.py`) implements a 6-phase systematic ablation protocol:

### Phase 1: Baseline Reproduction

5-seed evaluation of the unmodified system to establish baseline metrics:
```
μ_baseline = (1/S) Σ AUC^(s)
σ_baseline = sqrt((1/(S-1)) Σ (AUC^(s) − μ)²)
```

### Phase 2: Single-Factor Ablation

Each treatment T_i applied independently, evaluated with 5 seeds:
```
Δ_i = AUC(Baseline + T_i) − AUC(Baseline)
H₀: Δ_i ≤ 0  vs  H₁: Δ_i > 0
```

Statistical tests: DeLong's test (AUC comparison), McNemar's test (classification)
Correction: Holm-Bonferroni (m=7 multiple comparisons)

### Phase 3: Interaction Analysis

- **Pairwise**: `Interaction_{i,j} = Δ_{i+j} − (Δ_i + Δ_j)` → synergy/redundancy/independence
- **Cumulative forward selection**: Add treatments in rank order, measure marginal contribution

### Phase 4: Hyperparameter Sensitivity

Full HP grid scan for each significant treatment with 5-seed evaluation:
```
Sensitivity = std(AUC(θ)) / mean(AUC(θ))    (coefficient of variation)
```

Low sensitivity → robust (practical) | High sensitivity → fine-tuning needed

### Phase 5: Final Test Evaluation

One-time test set evaluation with the best treatment combination. No further tuning allowed after this phase.

### Phase 6: Feature Ablation Analysis

Four analyses for the LightGBM baseline:
1. Single-feature ablation
2. Static-attribute temporal aggregation validation
3. SHAP-based parsimonious model (top-k feature selection)
4. Logit pipeline integrity check

---

## 15. Usage Guide

### Full Pipeline

```bash
# Run everything: cache → detect fights → train → evaluate → report
python main.py --mode all --seed 42

# Quick test run with limited data
python main.py --mode all --max_matches 10 --seed 7
```

### Individual Pipeline Steps

```bash
# Cache building only
python main.py --mode build_cache

# Training only (assumes cache exists)
python main.py --mode train --models rnn_bigru,gnn_graphsage

# Specific models with custom configuration
python main.py --models lgbm,rnn_transformer,fusion_auto_best --seed 42
```

### Paper Presets

```bash
# 1-seed core preset: BiGRU + GraphSAGE + Transformer + 3-way Fusion
python runner.py --paper_preset core4_1seed --split_mode patch_holdout

# Fast triage preset (auto-caps max_matches at 600)
python runner.py --paper_preset core4_1seed_fast --paper_max_matches 600

# Optimal preset: BiGRU + GraphSAGE + event_xattn + Layered Fusion
python runner.py --paper_preset core4_optimal --split_mode patch_holdout
```

### Ablation Studies

```bash
# Phase 1: Baseline reproduction (5 seeds × baseline)
python experiment_runner.py --phase 1

# Phase 2: All single-factor treatments
python experiment_runner.py --phase 2 --treatment all

# Phase 2: Specific treatment only
python experiment_runner.py --phase 2 --treatment 1

# Phase 3: Interaction analysis (top-3 pairwise + forward selection)
python experiment_runner.py --phase 3

# Phase 4: HP sensitivity analysis
python experiment_runner.py --phase 4

# Phase 5: Final test evaluation (one-time)
python experiment_runner.py --phase 5

# Phase 6: Feature ablation analysis (SHAP, parsimonious model)
python experiment_runner.py --phase 6
```

### Fusion Sweeps

```bash
# Specific fusion combination
python runner.py --models rnn_bigru,gnn_graphsage \
    --fusion "rnn=bigru+gnn=graphsage+event=attn+logit=1"

# Full sweep: all RNN × all GNN × all event encoders
python runner.py --fusion "rnn_all+gnn_all+event_all+logit_all"
```

### Hardware Optimization

```bash
# Enable mixed precision + TF32 + speed profile
python runner.py --paper_preset core4_1seed --amp --tf32 --speed_profile rtx5080

# Enable torch.compile for kernel fusion
python runner.py --torch_compile --compile_mode max-autotune

# RAM caching for faster data loading
python runner.py --cache_match_packs_in_ram --cache_train_in_ram --cache_eval_in_ram

# Dataset sharing across models (eliminates repeated build_ms_sequence calls)
python runner.py --share_datasets
```

### Quality Reports

```bash
# Fight detection quality comparison across interpolation variants
python app/detection_quality_report.py \
    --variants full_interp,no_kill_traj_interp,snapshot_60s_no_interp \
    --reference full_interp --max-matches 200
```

---

## 16. Testing

### Test Suite

The project includes 14 test files with 234 test functions using pytest.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov

# Run a specific test file
pytest tests/test_teamfight_v2.py

# Run a specific test function
pytest tests/test_config.py::test_cfg_defaults
```

### Test Coverage

| Test File | Coverage Area |
|-----------|--------------|
| `test_common.py` | Safe numeric parsing, sigmoid/logit, sign-preserving log |
| `test_config.py` | CFG dataclass defaults, required fields, model list population |
| `test_fight_types.py` | FightRef construction, key stability/uniqueness, label boundary |
| `test_index_split.py` | Match-grouped splits, patch stratification, leakage checks |
| `test_experiment_runner.py` | Bootstrap CI calculation, seed determinism |
| `test_utils.py` | AUC, AP, Brier score, recall, confusion matrix |
| `test_teamfight_v2.py` | Kill clustering, spatial validation, fight classification |
| `test_interpolation.py` | Linear/cosine/exponential/cubic alpha curves |
| `test_postmerge_and_start_offset.py` | Post-merge spacing, START_OFFSET enforcement |
| `test_detection_quality_report.py` | Detection quality metrics and comparisons |
| `test_feature_ablation.py` | Feature contribution analysis, SHAP integration |
| `test_runner_paper_preset.py` | Paper preset CLI argument resolution |
| `test_speed_profile.py` | Speed profile configuration validation |

### Pytest Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

---

## 17. Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Kills-only fight creation** | Only CHAMPION_KILL events create fights — no ward/objective triggers |
| **XY-only interpolation** | Positions interpolated for spatial checks; all other features use piecewise-constant snapshots (ffill) |
| **No future leakage** | Node/global features from strictly-before 60s snapshots; label window starts at engage_ts |
| **XY excluded from model** | Player positions used only for fight detection, zeroed (`x_norm=0, y_norm=0`) in model input |
| **No double-counting** | Objectives/towers tracked only in post-fight outcome window, not in radius-3000 interaction count |
| **Millisecond precision** | All timestamps in ms; 5s grid enables sub-minute fight detection |
| **Ref-key alignment** | Predictions matched by `"match_id\|t_start_ts=<ms>"`, never by batch position |
| **Match-grouped splits** | All fights from one match stay in the same split (train/val/test) — prevents leakage |
| **Patch stratification** | Each split has proportional representation of game patches |
| **Deterministic seeding** | All randomness controlled by explicit seeds for full reproducibility |
| **Single source of truth** | All config in `core/config.py`; feature names in `core/feature_contract.py` |
| **Contract-based features** | Feature dimensions validated against contracts — detect misalignment early |
| **Calibration-aware** | Temperature scaling and ECE monitoring for reliable probability estimates |
| **Post-fight conversion** | 45-second window captures objective/tower/gold conversion after fight resolution |

---

## 18. Project Statistics

| Metric | Value |
|--------|-------|
| Total Python files | 92 |
| Total lines of code | ~31,000 |
| Test files | 14 |
| Test functions | 234 |
| Model architectures | 15+ variants |
| Configurable hyperparameters | 150+ |
| Node feature dimensions | 87 per player |
| Global feature dimensions | 27 per frame |
| Event feature dimensions | 48 per bin |
| Event token capacity | 64 tokens |
| Observation window | 60 seconds (12 × 5s bins) |
| Label window | 60 seconds |
| Default seeds | 5 (7, 42, 123, 256, 512) |
| Documentation files | 4 (this file + pipeline + paper + investigation) |
| Documentation lines | 4,000+ |

---

## Related Documents

- **[pipeline.md](pipeline.md)** — Detailed end-to-end pipeline walkthrough with diagrams
- **[CoG2026_Paper.md](CoG2026_Paper.md)** — Full IEEE CoG 2026 conference paper draft
- **[investigation_teamfight_duration.md](investigation_teamfight_duration.md)** — Data quality root-cause analysis
