# LOL Teamfight Lab

A research codebase for **predicting League of Legends teamfight outcomes** from Riot Match-V5 timeline data. Teamfight prediction is formulated as a **spatio-temporal multivariate time-series classification** task over heterogeneous player-interaction graphs: fights are detected from kill clusters with spatial validation, featurized into per-player / global / event streams, and classified by 25+ model architectures ranging from LightGBM to graph-temporal deep networks.

Companion code for our IEEE Conference on Games (CoG) 2026 paper:

> Seongeun Baek and Joonho Kwon, **"Kill-Conditioned Engagement Outcome Prediction in League of Legends Under Minute-Resolution Public Telemetry,"** IEEE Conference on Games (CoG), 2026.

---

## Highlights

- **Teamfight corpus construction** — a kill-cluster detection algorithm (temporal clustering + spatial diameter splitting + alive-player validation + adjacent-candidate merging) that builds ~1M validated engagements from ~200k ranked matches. No manual annotation required.
- **Multi-modal feature extraction** — 76-dim per-player node features, 26-dim global features, and 44-dim event aggregates over a 6-bin × 5-second pre-fight window, with strict no-future-leakage interpolation contracts.
- **25+ architectures under one harness** — LightGBM tabular baseline, recurrent (BiGRU/BiLSTM), attention (Transformer, EventXAttn), convolutional (TCN), state-space (Mamba), graph (GCN, GraphSAGE, GATv2, GraphTransformer, MPNN), spatio-temporal graph (ST-GNN, ST-GCN, EdgeSTGNN, ST-Mamba), and fusion/stacking ensembles.
- **7-treatment ablation protocol** — focal loss, game-phase encoding, attention pooling, momentum features, role-aware adjacency, multi-task learning, label smoothing; with bootstrap CIs and statistical testing.
- **Interpretability** — SHAP phase-stratified analysis, role/signal rollups, and optional LLM-assisted strategic interpretation.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/PIPELINE.md](docs/PIPELINE.md) | Complete 7-stage data pipeline from Riot API JSON to calibrated predictions |
| [docs/FEATURES.md](docs/FEATURES.md) | Exhaustive feature sets (76 node + 26 global + 44 event), dimensions, normalization |
| [docs/MODELS.md](docs/MODELS.md) | 25+ model architectures with mathematical definitions and hyperparameters |
| [docs/EXPERIMENT.md](docs/EXPERIMENT.md) | 7-treatment ablation protocol, statistical testing, evaluation metrics |
| [docs/CoG2026_Paper.md](docs/CoG2026_Paper.md) | Extended technical report (companion to the IEEE CoG 2026 paper) |
| [docs/AUDIT.md](docs/AUDIT.md) | Pre-submission code↔paper audit record |

---

## Pipeline Overview

```
Riot API JSONs -> Cache Build -> Fight Detection -> Index & Split
    -> Sample Construction (6 bins x 5s) -> Label Computation
    -> Model Training (25+ architectures) -> Ensemble Stacking
    -> Evaluation (AUC, AP, Brier, bootstrap CI)
```

1. **Detect teamfights** via kill-cluster temporal clustering with spatial validation (≥ 2 alive players per team within 1800 u; clusters split at 4000 u diameter; adjacent candidates merged within 15 s / 2000 u)
2. **Extract multi-modal features**: 76-dim per-player node features, 26-dim global features, 44-dim event aggregates
3. **Train 25+ architectures** with match-grouped, patch-stratified splits (random or patch-holdout)
4. **Ensemble predictions** through factorial stacking with meta-learner selection
5. **Ablate 7 domain-knowledge treatments** with seed-replicated runs and bootstrap confidence intervals

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for deep models; LightGBM runs on CPU)
- League of Legends match data collected via the [Riot API](https://developer.riotgames.com/) — Match-V5 **detail** and **timeline** JSON files

> **Note:** Raw match data is **not** included in this repository (Riot API Terms of Service). The full corpus is rebuilt deterministically from your own Match-V5 / Timeline records by the pipeline.

### Install

```bash
pip install -e ".[all]"      # everything (dev + analysis extras)
# or minimal:
pip install -r requirements.txt
```

### Configure Data Paths

Defaults are repo-relative (`data/raw/matches/kr/{detail,timeline}`, `outputs/`). Override with environment variables:

```bash
export LOL_DETAIL_DIR="/path/to/match/details"
export LOL_TIMELINE_DIR="/path/to/match/timelines"
export LOL_OUTPUT_ROOT="/path/for/outputs"
```

### Run

```bash
# Full pipeline (cache -> detect -> split -> train -> report)
python main.py --mode all --seed 42

# Paper presets
python runner.py --paper_preset core4_1seed --split_mode patch_holdout
python runner.py --paper_preset core4_optimal --split_mode patch_holdout

# Ablation studies
python experiment_runner.py --phase 1                  # Baseline reproduction
python experiment_runner.py --phase 2 --treatment all  # Single-factor ablations
```

### Test

```bash
pytest          # 376 tests
pytest --cov    # with coverage
```

---

## Project Structure

```
LOL_teamfight/
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
|   |-- pipeline.py                # Temporal feature building (6 bins x 5s)
|   |-- features.py                # Feature builders & normalizers
|-- train/                         # Model definitions & training
|   |-- models.py                  # 25+ model architectures
|   |-- model_registry.py          # Model key registry & factory
|   |-- deep.py                    # Deep learning training harness
|   |-- baseline.py                # LightGBM tabular baseline
|   |-- fusion.py                  # Ensemble stacking & fusion
|-- analysis/                      # SHAP, ablation reports, interpretation
|-- app/                           # Orchestration & analysis reporting
|-- tests/                         # Unit tests (376 cases)
|-- docs/                          # Documentation
```

---

## Reproducibility Note (relationship to the paper)

The metrics reported in the paper were generated with the experiment commit **prior to** the localization/labeling corrections recorded in [docs/AUDIT.md](docs/AUDIT.md). This released code implements the paper's described methods exactly:

- **Localization (Algorithm 1):** kill clusters are split by their true spatial *diameter* (> 4000 u), validity requires **≥ 2 *alive* players per team within 1800 u** of the earliest kill (a single conjunction), and Phase III **merges adjacent validated candidates within 15 s and 2000 u**.
- **Label (Eq. 3):** special-kill markers (ace / multi-kill / first-blood) contribute only the bonus `s(u)`, not a second kill.

On the corrected code the corpus is **994,365** validated engagements (about 4.8 per match) — the paper's "approximately one million" (the pre-correction construction was 1,115,123, ~5.4 per match). Each split (train / validation / test) is uniformly subsampled to **100,000 instances per seed** for training and evaluation. LightGBM reaches test **AUC ≈ 0.669** (seed 7, patch 15.16), about **0.006** below the **0.675** the paper reports and retains; this note is the record of the corrected figure. The relative ordering of paradigms (engineered tabular ≫ deep baselines) and all qualitative conclusions are unchanged.

---

## Citation

If you use this code in your research, please cite our IEEE CoG 2026 paper:

```bibtex
@inproceedings{baek2026killconditioned,
  author    = {Baek, Seongeun and Kwon, Joonho},
  title     = {Kill-Conditioned Engagement Outcome Prediction in League of
               Legends Under Minute-Resolution Public Telemetry},
  booktitle = {IEEE Conference on Games (CoG)},
  year      = {2026}
}
```

---

## Legal

This project is licensed under the [MIT License](LICENSE).

*LOL Teamfight Lab* is not endorsed by Riot Games and does not reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc.
