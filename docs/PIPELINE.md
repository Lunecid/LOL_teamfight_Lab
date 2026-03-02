# Data Pipeline: End-to-End Architecture

Complete specification of the seven-stage data pipeline from raw Riot API JSON to calibrated teamfight outcome predictions.

---

## Pipeline Overview

```
Riot API JSONs (match detail + timeline)
         |
   [Stage 1] Cache Build
         |  node_minute [T,10,F_node], global_minute [T,F_global],
         |  xy_raw_minute [T,10,2], events[], minute_ts[]
         v
   [Stage 2] Fight Detection (teamfight_v2)
         |  Kill clustering -> spatial validation -> FightRef generation
         v
   [Stage 3] Index & Split
         |  Match-grouped, patch-stratified partitioning
         |  Train (70%) / Val (20%) / Test (10%)
         v
   [Stage 4] Sample Construction
         |  12 bins x 5s observation window
         |  Node/Global/Event/Item feature tensors
         v
   [Stage 5] Label Computation
         |  y in {0,1} from kill_survival scoring
         |  Auxiliary regression targets
         v
   [Stage 6] Model Training & Inference
         |  15+ architectures: tabular, sequential, graph, fusion
         |  Ensemble stacking with meta-learner selection
         v
   [Stage 7] Evaluation & Reporting
         AUC, AP, Brier, ECE, minutewise, situation-aware metrics
         5-seed bootstrap CI, DeLong/McNemar significance tests
```

---

## Stage 1: Cache Build

**Entry point:** `data/cache_io.py::prebuild_cache()` -> `gameplay/pipeline.py::parse_timeline_to_minute_cache()`

Riot API provides timeline frames at **60-second intervals** plus raw events at **millisecond precision**. The cache pre-parses these into NumPy arrays stored as `.npz` files with JSON metadata.

### Cache Contents

| Field | Shape | Resolution | Description |
|-------|-------|------------|-------------|
| `node_minute` | `[T, 10, F_node]` | 60 s | Per-player feature vectors (F_node = 87) |
| `global_minute` | `[T, F_global]` | 60 s | Team-level aggregated features (F_global = 27) |
| `gold_team_minute` | `[T, 2]` | 60 s | Total gold per team (blue, red) |
| `xy_raw_minute` | `[T, 10, 2]` | 60 s | Raw player positions (x, y) in map units |
| `minute_ts` | `[T]` | 60 s | Frame timestamps in milliseconds |
| `events` | `List[dict]` | ms | All raw game events (kills, objectives, wards, items, etc.) |

Where `T` = number of 60-second frames in the match (typically 20-40 for a 20-40 minute game).

### Player Ordering Convention

Players are indexed 0-9 across all tensors:

| Index | Slot | Team | Role |
|-------|------|------|------|
| 0 | bTOP | Blue (100) | Top |
| 1 | bJNG | Blue (100) | Jungle |
| 2 | bMID | Blue (100) | Mid |
| 3 | bBOT | Blue (100) | Bot |
| 4 | bSUP | Blue (100) | Support |
| 5 | rTOP | Red (200) | Top |
| 6 | rJNG | Red (200) | Jungle |
| 7 | rMID | Red (200) | Mid |
| 8 | rBOT | Red (200) | Bot |
| 9 | rSUP | Red (200) | Support |

Role assignment uses `core/roles.py` based on Riot API `teamPosition` field.

### Cache Versioning

`CACHE_VERSION` (in `core/config.py`) is incremented whenever feature extraction logic changes. Cached files with stale versions are automatically invalidated and rebuilt.

---

## Stage 2: Fight Detection (teamfight_v2)

**Entry point:** `gameplay/fights.py::detect_fights()` -> `detect_fights_teamfight_v2()`

The detection algorithm uses a kills-only creation principle: only `CHAMPION_KILL` events create fights. No ward events, objective events, or multi-stage guards are used as triggers.

### Algorithm: Six Steps

```
Step 1: Build 5-Second Position Grid
Step 2: Cluster Kills by Temporal Proximity
Step 3: Validate Each Cluster as Teamfight
Step 4: Collect Interactions (radius 3000)
Step 5: Post-Fight Outcome (45-second window)
Step 6: Classify, Score, Output
```

### Step 1: Build 5-Second Position Grid

**Function:** `_build_5s_position_grid()`

Riot API provides player XY at 60-second intervals. For spatial checks we need finer resolution, so we interpolate to a **5-second dense grid**.

```
Riot API: 60s frames                  Dense 5s grid
  +- 0:00 -+- 1:00 -+- 2:00 -+        +- 0:00 - 0:05 - 0:10 - ... - 0:55 - 1:00 - 1:05 - ...
  | (x,y)  | (x,y)  | (x,y)  |  --->   |  interpolated XY at every 5-second mark
  +--------+--------+--------+        +- for all 10 players
```

**Layer 1 -- Baseline XY Interpolation:**

For each 5-second tick `t` between frame `F_i` (at `ts_i`) and `F_{i+1}` (at `ts_{i+1}`):

```
alpha_raw = (t - ts_i) / (ts_{i+1} - ts_i)       # alpha in [0, 1]
alpha = remap_alpha(alpha_raw, curve=INTERP_XY_CURVE)
XY(player, t) = (1 - alpha) * XY(F_i) + alpha * XY(F_{i+1})
```

Default curve: `exponential` with `k=3.0`, producing `alpha = 1 - exp(-k * alpha_raw)`.

**Layer 2 -- Pre-Kill Override:**

For each kill event (processed chronologically), override kill participants' positions:

```
Kill at ts=482000ms at position (8200, 4100)
  Participants: killer(pid=3), victim(pid=7), assists(pid=1, pid=4)

  For each participant:
    prior_frame = last 60s frame before kill
    override interval = [prior_frame_ts ... kill_ts]

    alpha_kill = (t - prior_frame_ts) / (kill_ts - prior_frame_ts)
    XY(participant, t) = (1 - alpha_kill) * XY(prior_frame) + alpha_kill * kill_position
```

Later kills overwrite earlier overrides. This grid is used **only for spatial checks** -- never as model input features.

### Step 2: Cluster Kills by Temporal Proximity

**Function:** `_cluster_kills_temporal(gap_ms=18000)`

Kills sorted by timestamp. Consecutive kills within **18 seconds** remain in the same cluster. When the gap exceeds 18s, a new cluster starts.

```
Timeline (ms):
  K1        K2    K3              K4   K5      K6
  |         |     |               |    |       |
  120000    125000 131000          180000 185000 192000
  |<--5s-->|<-6s->|               |<5s>|<-7s-->|
  |    within 18s gap              |   within 18s gap
  |                                |
  +------ Cluster A --------+    +------ Cluster B ------+
    first_kill: 120000               first_kill: 180000
    last_kill:  131000               last_kill:  192000
    center: K1 position              center: K4 position
                   ^ 49s gap ^
                  (> 18s -> split)
```

Each cluster produces:
- `first_kill_ts`, `last_kill_ts` -- temporal boundaries
- `fight_center` -- (x, y) of the first kill
- `participants` -- set of all killer/victim/assist participant IDs
- `n_kills` -- number of kills in the cluster

### Step 3: Validate Each Cluster as Teamfight

For each kill cluster:

```
3a. Compute engage time
    engage_ts = first_kill_ts - TF2_ENGAGE_PRE_KILL_MS (10,000 ms)
    (clamped to game start)

3b. Check context bounds
    engage_ts must be at least FIGHT_CONTEXT_MIN * 60000 ms into the game
    engage_ts + horizon must not exceed game end

3c. Check alive count
    At engage_ts: both teams must have >= TF2_MIN_PER_TEAM (2) alive champions

3d. SPATIAL VALIDATION: Radius 1800 check
    At engage_ts, look up all 10 positions from the 5s grid.
    Count how many are within radius = TF2_VALIDITY_RADIUS (1800) of fight_center.
    Require: >= 2 blue AND >= 2 red within radius.
```

```
        Radius 1800 check at engage_ts
        +----------------------------------+
        |                                  |
        |     *B1   *B2                    |
        |              + fight_center      |
        |     *R1   *R2                    |
        |                                  |   * = player inside
        +----------------------------------+   o = player outside
                                 oB3 oR3 oB4 oR4 oB5 oR5

  blue_in_radius = 2 (B1, B2)  >= 2  PASS
  red_in_radius  = 2 (R1, R2)  >= 2  PASS
  -> VALID TEAMFIGHT
```

If validation fails (e.g., only 1 player per team within radius), the cluster is rejected as a "pick" or isolated skirmish.

### Step 4: Collect Interactions (Radius 3000)

**Function:** `_collect_interactions_in_radius()`

Non-kill events during `[engage_ts, last_kill_ts]` within **radius 3000** of fight center are counted as fight interactions. Only position-based events (wards, summoner spells) are collected here.

**Important:** Objective events (`ELITE_MONSTER_KILL`, `BUILDING_KILL`, `TURRET_PLATE_DESTROYED`) are **NOT** counted as radius-3000 interactions. They are tracked only in the post-fight outcome window (Step 5). This prevents double-counting.

### Step 5: Post-Fight Outcome (45-second window)

**Function:** `_compute_postfight_outcome()`

After the last kill in the cluster, a **45-second window** captures consequences:

```
  --- fight ---                    --- post-fight window (45s) ---
  [engage ... last_kill]           [last_kill ... last_kill + 45000ms]
                    |                              |
                    |  Collect:                    |
                    |    * objectives taken         |
                    |    * towers destroyed         |
                    |    * gold differential        |
```

This captures whether the winning team converted kills into map objectives.

### Step 6: Classify, Score, Output

**Fight Type Classification** (`classify_fight_type()`):

| Condition (checked in order) | Classification |
|------------------------------|----------------|
| Center near Baron pit (< 1500 units) | `objective_baron` |
| Center near Dragon pit (< 1500 units) | `objective_dragon` |
| Center near Rift Herald (< 1500 units) | `objective_riftherald` |
| Center near a tower (< 1000 units) | `tower_dive` |
| Center near a base (< 3000 units) | `base_fight` |
| >= 8 proximity pairs | `teamfight` |
| >= 4 proximity pairs | `skirmish` |
| Otherwise | `pick` |

**Fight Outcome** (`compute_fight_outcome()`):

Counts kills, deaths, assists, gold swing, towers, and objectives in the label window `[engage_ts, horizon_end_ts)`.

### Detection Output Schema

```python
{
    "engage_ts":          int,    # ms -- fight start (primary temporal anchor)
    "horizon_end_ts":     int,    # ms -- label window end
    "first_kill_ts":      int,    # ms -- first kill in cluster
    "last_kill_ts":       int,    # ms -- last kill in cluster
    "centroid_x":         float,  # fight center X (from first kill)
    "centroid_y":         float,  # fight center Y (from first kill)
    "fight_type":         str,    # teamfight / skirmish / objective_baron / ...
    "outcome":            dict,   # kills, deaths, gold, towers per team
    "post_fight_outcome": dict,   # 45s window: objectives, towers, gold swing
}
```

### Detection Parameters (Complete)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `TF2_KILL_CLUSTER_GAP_MS` | 18,000 | ms | Max temporal gap between kills in same cluster |
| `TF2_ENGAGE_PRE_KILL_MS` | 10,000 | ms | Offset before first kill = engage time |
| `TF2_VALIDITY_RADIUS` | 1,800 | map units | Radius for teamfight validation (>= 2 per team) |
| `TF2_INTERACTION_RADIUS` | 3,000 | map units | Radius for counting fight interactions |
| `TF2_POST_FIGHT_WINDOW_MS` | 45,000 | ms | Post-fight outcome window duration |
| `TF2_MIN_PER_TEAM` | 2 | count | Minimum champions per team in validity radius |
| `FIGHT_MIN_GAP_MS` | 60,000 | ms | Minimum spacing between detected fights |
| `MAX_MERGED_FIGHT_DURATION_MS` | 120,000 | ms | Maximum allowed fight duration (reject if exceeded) |
| `FIGHT_CONTEXT_MIN` | 1 | minutes | Minimum game time before first fight |
| `FIGHT_HORIZON_SEC` | 60 | seconds | Label window duration |
| `CONTINUOUS_FIGHT_MAX_GAP_MS` | 30,000 | ms | Max gap for fight merging in post-merge |
| `CONTINUOUS_FIGHT_MERGE_RADIUS` | 2,000 | map units | Spatial threshold for merging |
| `START_OFFSET_MIN` | 2 | minutes | Minimum game-time offset (config-defined but not enforced in detection) |
| `DETECT_STEP_MS` | 10,000 | ms | Detection scanning step size |

### Coordinate System

- Map range: `[0, MAP_MAX]` where `MAP_MAX = 16,000` in both X and Y
- Normalized coordinates: `x_norm = x / MAP_MAX`, `y_norm = y / MAP_MAX`
- All radius checks use raw map units (not normalized)

---

## Stage 3: Index & Split

**Entry point:** `data/index_split.py::build_fight_index()` -> `split_refs()`

### FightRef Data Structure

Each detected fight becomes a `FightRef`:

```python
FightRef(
    match_id     = "KR_7123456789",
    patch        = "14.10",
    t_start      = 8,              # minute index
    t_start_ts   = 532000,         # engage timestamp in ms (primary key)
    label_end_ts = 592000,         # label window end in ms
)
```

**Primary key:** `ref_key = "KR_7123456789|t_start_ts=532000"`

This key is used throughout the pipeline for sample identification, prediction alignment, and leakage prevention.

### Split Strategies

| Mode | Description | Use Case |
|------|-------------|----------|
| `multi_patch` | Stratified by patch, grouped by match_id | Default -- balanced representation |
| `group_match` | Grouped by match_id only | When patch stratification is unnecessary |
| `patch_forward` | Train on older patches, test on newest | Temporal generalization evaluation |
| `patch_holdout` | Specific patches held out for testing | Controlled patch-transfer experiments |
| `random` | Random stratified split | Baseline / sanity check |

### Split Ratios

| Split | Default Ratio | Purpose |
|-------|---------------|---------|
| Train | 70% | Model training |
| Validation | 20% | Model selection, early stopping, hyperparameter tuning |
| Test | 10% | Final evaluation (never used for model selection) |

### Leakage Prevention

**Match-grouped splitting:** All fights from one match stay in the same split partition. This prevents a model from memorizing match-specific patterns (player identities, team compositions, gold trajectories) during training and exploiting them during evaluation.

**Patch-stratified:** Each split has proportional representation of game patches, preventing distributional mismatch between training and evaluation.

---

## Stage 4: Sample Construction (Observation Window)

**Entry point:** `gameplay/pipeline.py::build_ms_sequence()`

### Timeline Layout

```
  <------------ observation window (60s) ------------->
  |                                                    |
  |  bin0   bin1   bin2   ...   bin10  bin11            |
  | [0-5s] [5-10s] [10-15s]         [50-55s] [55-60s] |
  |                                                    |
  start_ms                                          end_ms = engage_ts
  (engage_ts - 60s)                                    |
                                                       |
                                       <--- label window (60s) --->
                                       |                           |
                                   engage_ts               label_end_ts
                                  (= fight start)      (= engage + horizon)
                                       |                           |
                                       |   kills, deaths, gold     |
                                       |   counted here -> label   |
```

**Key principle:** The model sees 60 seconds of game state **before the fight starts**. It never sees the fight outcome -- that is the label.

### Time Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CONTEXT_MS` (ctx_ms) | 60,000 ms | Observation window duration |
| `BIN_MS` (bin_ms) | 5,000 ms | Time bin size |
| `HORIZON_MS` (horizon_ms) | 60,000 ms | Label window duration |
| `PREDICTION_GAP_MS` | 0 ms | Gap between observation end and engage_ts |
| **Derived: L** | **12** | Number of temporal bins = ctx_ms / bin_ms |

### Per-Bin Computation

For each of the 12 bins, the midpoint timestamp `q` is computed:

```
bin_i:  b0 = start_ms + i * 5000
        b1 = start_ms + (i+1) * 5000
        q  = b0 + 2500  (midpoint)
```

At each midpoint `q`:

| Tensor | Source | Method | Shape |
|--------|--------|--------|-------|
| `node_i` | 60s frame strictly before `q` | Piecewise-constant (step-hold / ffill) | `(10, F_node)` |
| `glob_i` | 60s frame strictly before `q` | Piecewise-constant (step-hold / ffill) | `(F_global,)` |
| `ev_i` | Events in `[b0, b1)` | Bin-level aggregation (count) | `(F_event,)` |
| `item_i` | Item purchases in `[b0, b1)` | Hash encoding | `(F_item,)` |

### Feature Handling Rules

```
+---------------------------------------------------------------+
|  RULE: "XY interpolation only; all other features use snapshots"  |
|                                                                    |
|  1. Fight Detection (5s grid):                                     |
|     XY IS interpolated -- dense 5s grid for radius checks          |
|     Internal to detect_fights_teamfight_v2()                       |
|                                                                    |
|  2. Model Input -- Node/Global (observation window):               |
|     NO interpolation of scalar features                            |
|     Use strict-before 60s snapshot (piecewise-constant / ffill)    |
|     INTERP_SCALARS_METHOD = "ffill"                                |
|                                                                    |
|  3. Model Input -- XY:                                             |
|     XY is ZEROED (x_norm=0, y_norm=0 in every bin)                |
|     Prevents model from memorizing map-position bias               |
|     Config: ZERO_XY_NODE_FEATURES = True                           |
|                                                                    |
|  4. Events/Items:                                                  |
|     Bin-level aggregation [b0, b1) -- NOT interpolation            |
|     Counts raw events that occurred in the bin time interval       |
+---------------------------------------------------------------+
```

### Output Tensors

```python
sample = {
    "node_seq":  np.array[12, 10, F_node],   # per-player features, 12 bins
    "glob_seq":  np.array[12, F_global],      # team-level features, 12 bins
    "ev_seq":    np.array[12, F_event],       # event aggregations, 12 bins
    "item_seq":  np.array[12, F_item],        # item hashes, 12 bins
    "y":         int,                          # label: 1=blue wins, 0=red wins
}
```

### Interpolation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INTERP_XY_METHOD` | `"linear_guard_midstep"` | XY interpolation method for position grid |
| `INTERP_XY_CURVE` | `"exponential"` | Curve shape (k=3.0) |
| `INTERP_SCALARS_METHOD` | `"ffill"` | Scalar feature interpolation (= no interpolation) |
| `ZERO_XY_NODE_FEATURES` | `True` | Zero XY in model node features |
| `ZERO_XY_IN_EXTRA_SEQ` | `True` | Zero XY in extra sequence (for RNN) |
| `USE_RELATIVE_XY` | `True` | Use centroid-relative coordinates |

---

## Stage 5: Label Computation

**Entry point:** `gameplay/pipeline.py::compute_label_targets()`

### Primary Label: `kill_survival` (default)

Events within the label window `[engage_ts, engage_ts + horizon_ms)` determine the fight outcome:

```
Score = W_KILL * (blue_kills - red_kills) + W_ALIVE * (blue_alive - red_alive)

W_KILL  = 1.0   (kill differential weight)
W_ALIVE = 0.3   (alive-at-end differential weight)

y = 1  if Score > 0   -> blue team wins the fight
y = 0  if Score < 0   -> red team wins the fight
tie -> LABEL_TIE_STRATEGY = "random" (seeded for reproducibility)
```

**Example:**

```
Label window [420000, 480000]:
  Blue kills: 3, Red kills: 1  ->  kill_diff = +2
  Blue alive at end: 4, Red alive at end: 2  ->  alive_diff = +2

  Score = 1.0 * 2 + 0.3 * 2 = 2.6 > 0  ->  y = 1 (blue wins)
```

### Auxiliary Targets (Multi-Task Learning)

| Target | Source Window | Normalization | Description |
|--------|-------------|---------------|-------------|
| `y_kill_diff` | Fight window | kill_diff / 5.0 | Normalized kill differential |
| `y_gold_diff` | Fight window | gold_diff / 1000.0 | Normalized gold swing |
| `y_obj_diff` | Fight window | obj_diff / 5.0 | Normalized objective differential |
| `y_alive_diff_raw` | Fight window | Raw count | Alive count differential |
| `post_gold_diff` | 45s outcome | Raw gold | Gold swing after fight |
| `post_tower_diff` | 45s outcome | Raw count | Towers taken after fight |
| `post_obj_diff` | 45s outcome | Raw count | Objectives taken after fight |

### Label Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LABEL_MODE` | `"kill_survival"` | Label computation method |
| `W_KILL` | 1.0 | Kill differential weight |
| `W_ALIVE` | 0.3 | Alive count weight |
| `LABEL_TIE_STRATEGY` | `"random"` | Tie handling: `"random"`, `"drop"`, `"blue"`, `"red"` |

---

## Stage 6: Model Training & Inference

**Entry point:** `app/experiment.py` -> `train/deep.py`, `train/baseline.py`, `train/fusion.py`

### Training Loop

1. **Data loading:** `InMemoryFightDataset` pre-loads all samples into RAM. `collate_batch()` constructs graph adjacency matrices per batch.
2. **Forward pass:** Model receives `(node_seq, extra_seq, y)` plus optional event tokens.
3. **Loss computation:** BCE (default) or Focal Loss + optional multi-task auxiliary losses.
4. **Optimization:** AdamW with gradient clipping and optional mixed precision (AMP).
5. **Early stopping:** Monitor validation AUC with patience.
6. **Prediction alignment:** Predictions are aligned by `ref_key` (not batch position) to handle shuffling.

### Ensemble Stacking

Three meta-learning strategies combine base model predictions:

| Strategy | Description |
|----------|-------------|
| **Simple stacking** | Logistic regression meta-learner on train-split logits |
| **Out-of-fold (OOF)** | K-fold (K=5) cross-validation for unbiased meta-features |
| **Factorial stacking** | Enumerate all 2^M - 1 subsets of M base models, select best by val AUC |

After model selection, the meta-learner is refit on train+val combined.

### Per-Patch Temperature Scaling

```
P_calibrated = sigmoid(z / T_p*)

T_p* = argmin_T  sum_i [ -y_i log sigmoid(z_i/T) - (1-y_i) log(1 - sigmoid(z_i/T)) ]
```

---

## Stage 7: Evaluation & Reporting

**Entry point:** `app/experiment.py`, `app/analysis_reporting.py`

### Prediction Alignment

```python
for batch in dataloader:
    ref_keys = batch["ref_key"]       # List[str]
    logits   = model(batch)           # (B, 1)
    for key, logit in zip(ref_keys, logits):
        logit_map[key] = sigmoid(logit)
```

### Metrics (see docs/EXPERIMENT.md for full definitions)

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under ROC Curve (primary metric) |
| **AP** | Average Precision |
| **Accuracy** | At threshold 0.5 |
| **Precision / Recall / F1** | Per-class and macro |
| **Brier Score** | Calibration: (1/N) sum (p_i - y_i)^2 |
| **ECE** | Expected Calibration Error |

### Subgroup Analysis

- **By game minute:** early (< 15 min), mid (15-25 min), late (> 25 min)
- **By gold state:** close (|gold_diff| < 2000), moderate, stomp (|gold_diff| > 5000)
- **By fight type:** teamfight vs. skirmish vs. objective contest
- **By patch:** per-patch performance tracking

### Statistical Testing

- **Bootstrap CI:** 5 seeds {7, 42, 123, 256, 512}, 1000 resamples, percentile method
- **DeLong's test:** AUC comparison (correlated samples)
- **McNemar's test:** Classification disagreement (continuity-corrected)
- **Holm-Bonferroni:** Multiple comparison correction (alpha = 0.05, m = 7 treatments)

---

## Complete Timeline Diagram

```
  Game Timeline (ms)
  =====================================================================

  0        60000      120000     180000     240000     300000
  +----------+----------+----------+----------+----------+
  |          |          |          |          |          |
  |  60s frame snapshots from Riot API (node_minute)    |
  |                                                     |
  |                    Kill K1 ----+                     |
  |                    Kill K2 --+ | < 18s gap           |
  |                    Kill K3 + | |  (same cluster)     |
  |                            | | |                     |
  |                            v v v                     |
  |                     +------ Kill Cluster -----+     |
  |                     | first: K1               |     |
  |                     | last:  K3               |     |
  |                     | center: K1 (x,y)        |     |
  |                     +-------+-----------------+     |
  |                             |                       |
  |               +-------------+                       |
  |               v             v                       |
  |          engage_ts    first_kill_ts                  |
  |          (K1 - 10s)   (K1)                          |
  |               |                                     |
  |     +---------+                                     |
  |     | Radius  |                                     |
  |     | 1800    | >= 2 blue + >= 2 red? -> VALID      |
  |     | check   |                                     |
  |     +---------+                                     |
  |               |                                     |
  |               |<--- observation window (60s) --->|  |
  |               |                                  |  |
  |               |<--- label window (60s) --------->|  |
  |               |     [engage_ts, +60s]            |  |
  |               |                            last_kill_ts
  |               |                                  |  |
  |               |                                  |<-- 45s -->|
  |               |                                  | post-fight |
  |               |                                  | outcome    |
  |                                                     |
  =====================================================================
```

---

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Kills-only creation** | Only `CHAMPION_KILL` events create fights -- no ward/objective triggers |
| **Snapshot features (no interpolation)** | Node/global use strict-before 60s snapshot (ffill); only XY is interpolated (for spatial checks) |
| **No future leakage** | Node + global features from strictly-before snapshots; label window starts at engage_ts |
| **XY excluded from model** | Positions used only for spatial detection and adjacency, zeroed in model input |
| **No double-counting** | Objectives/towers tracked only in post-fight outcome (Step 5), not in radius-3000 interactions (Step 4) |
| **Millisecond anchoring** | All timestamps in ms; sub-minute fight precision via 5s grid |
| **Ref-key alignment** | Predictions matched by `match_id|t_start_ts=<ms>`, not by batch position |
| **Match-grouped splits** | All fights from one match stay in the same split partition |
| **Patch stratification** | Each split has proportional representation of game patches |
| **Post-fight conversion** | 45-second window captures objective/tower/gold conversion after fight |

---

## Full End-to-End Example

```
MATCH: KR_7123456789, Patch 14.10, Duration 32:00
==========================================================

[1] Cache Build
    -> 33 minute frames (0:00 -> 32:00)
    -> node_minute: [33, 10, 87]
    -> 847 raw events (kills, objectives, wards, ...)

[2] Fight Detection (teamfight_v2)
    -> Extract 28 CHAMPION_KILL events
    -> Cluster temporally (gap=18s): 6 clusters
    -> Build 5s position grid: [385 timesteps, 10, 2]
    -> Validate each cluster:
        Cluster 1 (3 kills, 7:10-7:25): 3v3 at radius 1800 -> valid teamfight
        Cluster 2 (1 kill, 11:40):      1v1 at radius 1800 -> rejected (pick)
        Cluster 3 (4 kills, 15:20-15:45): 4v4 at radius 1800 -> valid teamfight
        Cluster 4 (2 kills, 20:05-20:12): 2v3 at radius 1800 -> valid teamfight
        Cluster 5 (5 kills, 26:30-27:00): 5v5 at radius 1800 -> valid teamfight
        Cluster 6 (2 kills, 31:10-31:15): horizon exceeds game -> rejected
    -> 4 fights detected

[3] FightRef Index
    -> Fight 1: "KR_7123456789|t_start_ts=420000"
    -> Fight 3: "KR_7123456789|t_start_ts=910000"
    -> Fight 4: "KR_7123456789|t_start_ts=1195000"
    -> Fight 5: "KR_7123456789|t_start_ts=1580000"
    -> Split: match grouped into "train" partition

[4] Sample Build (Fight 1: engage_ts = 420000)
    -> Observation window: [360000, 420000] (6:00 -> 7:00)
    -> 12 bins x 5s each
    -> Per bin: snapshot node+global (strict-before 60s frame),
               aggregate events, hash items
    -> XY zeroed in all bins
    -> node_seq: [12, 10, 87], glob_seq: [12, 27], ev_seq: [12, 48]

[5] Label (Fight 1)
    -> Label window: [420000, 480000] (7:00 -> 8:00)
    -> Events in window: 3 blue kills, 1 red kill
    -> Blue alive at 8:00: 4, Red alive: 2
    -> Score = 1.0 * 2 + 0.3 * 2 = 2.6 -> y = 1 (blue wins)

[6] Model Prediction
    -> GATv2 forward: node_seq + extra -> logit = 1.34
    -> P(blue wins) = sigmoid(1.34) = 0.79
    -> Prediction: blue team wins (confidence 79%)

[7] Evaluation
    -> Aligned by ref_key: "KR_7123456789|t_start_ts=420000"
    -> True label: y=1, Predicted: 0.79 -> correct
    -> Contributes to AUC, AP, accuracy metrics
```
