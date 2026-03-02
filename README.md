# LOL Teamfight Lab

A machine learning pipeline for predicting **League of Legends teamfight outcomes** using match timeline data from the Riot API. The project formulates teamfight prediction as a **spatio-temporal multivariate time-series classification** task over heterogeneous player interaction graphs.

**Target Venue:** IEEE Conference on Games (CoG) 2026

---

## Documentation

| Document | Description |
|----------|-------------|
| **[docs/PIPELINE.md](docs/PIPELINE.md)** | Complete 7-stage data pipeline from Riot API JSON to calibrated predictions |
| **[docs/FEATURES.md](docs/FEATURES.md)** | Exhaustive feature sets (87 node + 27 global + 48 event), dimensions, normalization |
| **[docs/MODELS.md](docs/MODELS.md)** | 15+ model architectures with mathematical definitions and hyperparameters |
| **[docs/EXPERIMENT.md](docs/EXPERIMENT.md)** | 7-treatment ablation protocol, statistical testing, evaluation metrics |
| **[docs/CoG2026_Paper.md](docs/CoG2026_Paper.md)** | Full paper draft (IEEE CoG 2026) |

---

## Pipeline Overview

```
Riot API JSONs -> Cache Build -> Fight Detection -> Index & Split
    -> Sample Construction (12 bins x 5s) -> Label Computation
    -> Model Training (15+ architectures) -> Ensemble Stacking
    -> Evaluation (AUC, AP, Brier, bootstrap CI)
```

1. **Detect teamfights** via kill-cluster-based temporal clustering with spatial validation (radius 1800, >= 2 per team)
2. **Extract multi-modal features**: 87-dim per-player node features, 27-dim global features, 48-dim event aggregates
3. **Train 15+ architectures**: LightGBM, BiGRU, Transformer, TCN, Mamba, GCN, GraphSAGE, GATv2, MPNN, ST-GNN, EventXAttn, Layered Fusion
4. **Ensemble predictions** through factorial stacking with meta-learner selection
5. **Ablate 7 domain-knowledge improvements**: focal loss, game phase encoding, attention pooling, momentum features, role-aware adjacency, multi-task learning, label smoothing

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- League of Legends match data in JSON format (detail + timeline files)

### Install

```bash
pip install -e ".[all]"
# or
pip install -r requirements.txt
```

### Configure Data Paths

```bash
export LOL_DETAIL_DIR="/path/to/match/details"
export LOL_TIMELINE_DIR="/path/to/match/timelines"
export LOL_OUTPUT_ROOT="/path/for/outputs"
```

### Run

```bash
# Full pipeline
python main.py --mode all --seed 42

# Paper presets
python runner.py --paper_preset core4_1seed --split_mode patch_holdout
python runner.py --paper_preset core4_optimal --split_mode patch_holdout

# Ablation studies
python experiment_runner.py --phase 1   # Baseline reproduction
python experiment_runner.py --phase 2 --treatment all  # Single-factor
```

---

## Project Structure

```
LOL_teamfight_Lab/
|-- main.py / runner.py            # Entry points
|-- experiment_runner.py           # Ablation study runner
|-- core/                          # Configuration, contracts, utilities
|   |-- config.py                  # Central CFG dataclass (single source of truth)
|   |-- feature_contract.py        # Feature name/dimension contracts
|   |-- improvements.py            # 7 domain-knowledge treatments (T1-T7)
|-- data/                          # Data loading, caching, splitting
|   |-- cache_io.py                # Match cache I/O (NPZ + JSON)
|   |-- dataset.py                 # InMemoryFightDataset (PyTorch)
|   |-- index_split.py             # Match-grouped, patch-stratified splits
|-- gameplay/                      # Fight detection & feature engineering
|   |-- fights.py                  # teamfight_v2 detection algorithm
|   |-- pipeline.py                # Temporal feature building (12 bins x 5s)
|   |-- features.py                # Feature builders & normalizers
|-- train/                         # Model definitions & training
|   |-- models.py                  # 15+ model architectures
|   |-- deep.py                    # Deep learning training harness
|   |-- baseline.py                # LightGBM tabular baseline
|   |-- fusion.py                  # Ensemble stacking & fusion
|   |-- graph_encoder.py           # GNN encoder implementations
|   |-- temporal_encoders.py       # RNN/Transformer/TCN/Mamba encoders
|-- app/                           # Orchestration & analysis
|   |-- experiment.py              # Main training loop orchestrator
|-- tests/                         # Unit tests (100+ cases)
|-- docs/                          # Documentation
```

---

## Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch (CUDA, AMP, torch.compile) |
| **Gradient Boosting** | LightGBM |
| **Numerical** | NumPy |
| **ML Utilities** | scikit-learn |
| **Hardware** | CUDA GPU, TF32 on Ampere, bf16/fp16 mixed precision |

---

## Testing

```bash
pytest                    # Run all tests
pytest --cov              # With coverage
pytest tests/test_utils.py  # Specific file
```

---

## License

This project is for research and educational purposes.
