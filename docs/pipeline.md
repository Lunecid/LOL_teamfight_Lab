# Pipeline: Detect & Predict

End-to-end data flow from Riot API timeline JSON to teamfight outcome prediction.

```
  Riot API Timeline JSON
          |
          v
  [1] Cache Build ──────────── node_minute, global_minute, events
          |
          v
  [2] Fight Detection ──────── engage_ts, centroid, participants
          |
          v
  [3] FightRef Index ────────── match_id | t_start_ts=<ms>
          |
          v
  [4] Sample Build ─────────── node_seq, glob_seq, ev_seq, item_seq
          |
          v
  [5] Feature Extraction ───── spatial, momentum, game-phase features
          |
          v
  [6] Label Computation ────── y ∈ {0,1}, auxiliary regression targets
          |
          v
  [7] Tensor Collation ─────── batched tensors (B, L, N, F)
          |
          v
  [8] Model Forward ────────── logit → sigmoid → prob ∈ [0,1]
          |
          v
  [9] Evaluation ───────────── AUC, AP, Accuracy (ref_key aligned)
```

---

## 1. Cache Build

**Entry:** `data/cache_io.py::prebuild_cache()` → `gameplay/pipeline.py::parse_timeline_to_minute_cache()`

Riot API provides timeline frames at **60-second intervals** plus raw events at **millisecond precision**. The cache pre-parses these into NumPy arrays.

### Cache Contents

| Field | Shape | Resolution | Description |
|-------|-------|------------|-------------|
| `node_minute` | `[T, 10, F_node]` | 60s | Per-player feature vectors |
| `global_minute` | `[T, F_global]` | 60s | Team-level aggregated features |
| `gold_team_minute` | `[T, 2]` | 60s | Total gold per team |
| `xy_raw_minute` | `[T, 10, 2]` | 60s | Raw player positions |
| `minute_ts` | `[T]` | 60s | Frame timestamps in ms |
| `events` | `List[dict]` | ms | All raw game events |

### Node Features (per player, per frame)

```
Position:      x_norm, y_norm
Resources:     level_norm, xp_norm, curGold_norm, totalGold_norm, gps_norm
CS:            laneCS_norm, jgCS_norm
Status:        alive (0/1), hp_pct, mp_pct
Identity:      champion_id (categorical)
Runes:         primary_rune_1-4, sub_rune_1-2 (categorical)
Buffs:         has_baron, has_elder, has_red, has_blue (0/1)
Buff Duration: baron_remain_norm, elder_remain_norm, ...
Cooldowns:     ult_level_norm, flash_ready, flash_remain_norm
Vision:        ward_count, ward_kills, vision_score (normalized)
Champion Stats: armor, AD, AP, MR, AS (25 features, normalized)
Damage Stats:  total/magic/physical damage to champs (12 features)
Dragon Soul:   soul_infernal, soul_ocean, soul_mountain, soul_cloud, soul_hextech
```

### Global Features (per frame)

```
Differentials: gold_diff, xp_diff, level_diff, cs_diff, alive_diff
Cumulative:    kill_diff, tower_diff, inhib_diff, dragon_diff, baron_diff
Bans:          blue_ban_0-4, red_ban_0-4 (champion IDs)
Time:          time_norm ∈ [0, 1]
```

---

## 2. Fight Detection

**Entry:** `gameplay/fights.py::detect_fights()`

Three detection algorithms are available, selected by `FIGHT_DETECT_ALGO` config:

### Algorithm: `killchain_v1` (kill-chain based)

Uses `victimDamageReceived[]` from `CHAMPION_KILL` events to identify all combatants at millisecond precision, then chains overlapping kills into fights.

```
Step 1: Extract rich kill events
  CHAMPION_KILL → {timestamp, killer, victim, assists, position, damageReceived[]}

Step 2: Get all participants per kill
  victimDamageReceived[] → all pids who dealt damage
  Union with killer + victim + assists

Step 3: Chain kills (Union-Find)
  Kill_A shares participants with Kill_B within 30s → same chain

Step 4: Chain → Fight candidate
  engage_ts = first_kill_ts - 10s (backtrack)
  centroid  = mean(kill positions)
  participants = union of all damage arrays

Step 5: Filter
  ≥ 2 participants per team (from damage arrays)
  Within match time bounds

Step 6: Post-process
  ST-DBSCAN merge nearby candidates
  ACE truncation
  Spacing enforcement (≥ 120s gap)

Step 7: Classify + compute outcome
  Near dragon pit → "objective_dragon"
  Near tower      → "tower_dive"
  8+ participants → "teamfight"
  Outcome: kills, gold swing per team
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KILLCHAIN_WINDOW_MS` | 30,000 | Max time gap to chain two kills |
| `KILLCHAIN_BACKTRACK_MS` | 10,000 | How far before first kill to set engage_ts |
| `fight_min_gap_ms` | 60,000 | Minimum spacing between fights |
| `continuous_fight_merge_radius` | 2,000 | Spatial merge radius |

### Algorithm: `event_v1` (event-driven)

Scores event bursts (kills, spells, objectives, buildings) within sliding windows to identify fight candidates.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVENT_BURST_WINDOW_MS` | 15,000 | Sliding window size |
| `EVENT_SCORE_THRESHOLD` | 2.5 | Min score to trigger candidate |
| `EVENT_WEIGHT_KILL` | 2.0 | Kill event weight |
| `EVENT_WEIGHT_SPELL` | 0.35 | Spell event weight |
| `EVENT_WEIGHT_OBJECTIVE` | 1.5 | Objective event weight |
| `EVENT_WEIGHT_BUILDING` | 1.5 | Building event weight |

### Algorithm: `engage_v2` (position-based)

Dense XY interpolation to detect standoff-to-engagement transitions via distance drops between teams.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `standoff_radius` | 1,800 | Engagement distance threshold |
| `standoff_min_pairs` | 3 | Min proximity pairs |
| `engage_min_dist_drop` | 250 | Min distance drop for engagement |
| `detect_step_ms` | 10,000 | Detection step size |

### Output

Each detected fight produces:

```python
{
    "engage_ts":     int,    # ms — when fight starts (primary anchor)
    "horizon_end_ts": int,   # ms — label window end
    "centroid_x":    float,  # mean fight position X
    "centroid_y":    float,  # mean fight position Y
    "fight_type":    str,    # teamfight / skirmish / pick / tower_dive / objective_*
    "outcome":       dict,   # kills, deaths, gold per team
}
```

---

## 3. FightRef Index

**Entry:** `data/index_split.py::build_fight_index()`

Each detected fight becomes a `FightRef` — the unique identifier that tracks a sample through the entire pipeline.

```python
FightRef(
    match_id    = "KR_7123456789",
    patch       = "14.10",
    t_start     = 8,            # minute index (legacy)
    t_start_ts  = 532000,       # engage timestamp in ms (primary)
    label_end_ts = 592000,      # label window end in ms
)
```

**Primary key:** `ref_key = "KR_7123456789|t_start_ts=532000"`

### Split Strategies

| Mode | Description |
|------|-------------|
| `multi_patch` | Stratified by patch, grouped by match_id (default) |
| `group_match` | Grouped by match_id only |
| `patch_forward` | Train on old patches, test on newest |
| `random` | Random stratified split |

Default split ratios: **70% train / 20% val / 10% test**

---

## 4. Sample Build

**Entry:** `gameplay/pipeline.py::build_ms_sequence()`

Given a `FightRef`, builds the observation window and label window.

### Timeline Layout

```
                    ctx_ms (60s)
              |◄─────────────────────►|
              |   12 bins × 5s each   |
              |                       |
  [start_ms ........................ end_ms]
              |   observation window   |
                                      ↑
                                  engage_ts
                                      |
                                      |◄── horizon_ms (60s) ──►|
                                      |    label window         |
                                      [engage_ts ...... label_end_ts]
```

### Time Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ctx_ms` | 60,000 | Observation window duration |
| `bin_ms` | 5,000 | Time bin size |
| `horizon_ms` | 60,000 | Label window duration |
| `prediction_gap_ms` | 0 | Gap between observation end and engage_ts |

**Derived:** `L = ctx_ms / bin_ms = 12 time steps`

### Per-Bin Computation

For each of the 12 time bins:

```
q = bin midpoint (ms)

node_i  = interpolate_node_global(cache, q)
  → Linear interpolation between 60s frames
  → XY uses jump detection + midstep snapping
  → Zeroed if ZERO_XY_NODE_FEATURES = True

glob_i  = global_from_prev_snapshot(cache, q)
  → Nearest frame strictly before q (no future leakage)

ev_i    = aggregate_events(cache, team_map, bin_start, bin_end)
  → Event counts/features within [b0, b1)

item_i  = aggregate_items(cache, team_map, bin_start, bin_end)
  → Item purchase hashes within [b0, b1)
```

### Raw Output

```python
{
    "node_seq":  np.array[L, 10, F_node],    # per-player features
    "glob_seq":  np.array[L, F_global],       # team-level features
    "ev_seq":    np.array[L, F_event],        # event aggregations
    "item_seq":  np.array[L, F_item],         # item hashes
}
```

---

## 5. Feature Extraction

**Entry:** `gameplay/features.py::build_sequence_features()`

Transforms raw sequences into model-ready feature tensors. Five feature sets are available:

| Feature Set | Components | Use Case |
|-------------|------------|----------|
| `full` | node_seq + macro + spatial | GNN / RNN models (default) |
| `global_only` | global base + spatial | Lightweight baseline |
| `global_events` | global + events + spatial | Event-focused models |
| `node_personal` | per-node + minimal events + spatial | Per-champion analysis |
| `tri_modal` | node + macro + tabular static | Fusion models |

### Spatial Features (23 dimensions)

Computed from player XY positions at each time step:

```
Team Centroids:    pos_fight_x/y, pos_blue_x/y, pos_red_x/y
Separation:        dist_team_sep_norm
Engagement:        standoff_pairs_frac, mean_min_enemy_dist_norm
Deltas:            d_standoff_pairs_frac, d_mean_min_enemy_dist_norm
Objectives:        dist_obj_nearest, near_obj_dragon/baron/herald/atakhan/horde
Structures:        dist_tower_nearest, in_tower_range, near_tower_radius
Map Zones:         zone_top_lane, zone_mid_lane, zone_bot_lane, zone_river, zone_jungle
```

---

## 6. Label Computation

**Entry:** `gameplay/pipeline.py::compute_label_targets()`

Counts events within the label window `[engage_ts, engage_ts + horizon_ms)` to determine fight outcome.

### Label Types

**`kill_survival` (default):**

```
score = W_KILL × kill_diff + W_ALIVE × alive_diff
y = 1  if score > 0 (blue wins)
y = 0  if score < 0 (red wins)
ties → TIE_POLICY (drop by default)
```

| Weight | Default | Description |
|--------|---------|-------------|
| `LABEL_W_KILL` | 1.0 | Kill differential weight |
| `LABEL_W_ALIVE` | 0.3 | Alive-at-end differential weight |

**`micro_win`:**

```
score = blue_kills - red_kills
y = 1 if score > 0, else y = 0
```

### Auxiliary Targets (multi-task learning)

| Target | Normalization | Description |
|--------|---------------|-------------|
| `y_kill_diff` | kill_diff / 5.0 | Normalized kill differential |
| `y_gold_diff` | gold_diff / 1000.0 | Normalized gold swing |
| `y_objective_diff` | obj_diff / 5.0 | Normalized objective differential |
| `y_tower_diff` | tower_diff / 5.0 | Normalized tower differential |
| `y_alive_diff_raw` | raw count | Alive count differential |

---

## 7. Tensor Collation

**Entry:** `data/dataset.py::InMemoryFightDataset` → `collate_batch()`

Three tensor layouts depending on model architecture:

### Layout A: GNN / RNN (default)

```
node_seq:  (B, L, 10, F_node)   float32   per-player temporal
extra_seq: (B, L, D_extra)      float32   macro + spatial
y:         (B, 1)               float32   binary label
```

### Layout B: MacroFusion

```
node_seq:  (B, L, 10, F_node)   float32   per-player temporal
macro_seq: (B, L, D_macro)      float32   team-level temporal
tab_x:     (B, D_tab)           float32   static match features
y:         (B, 1)               float32   binary label
```

### Layout C: Flat Sequence

```
x_seq:     (B, L, D_flat)       float32   flattened features
y:         (B, 1)               float32   binary label
```

### Optional Additions

```
Event tokens (cross-attention):
  event_type:  (B, K, int64)      event type hash
  event_actor: (B, K, int64)      participant ID
  event_team:  (B, K, int64)      0=blue, 1=red, 2=unknown
  event_cont:  (B, K, 5, float32) [t_rel, dt_end, x, y, val]
  event_mask:  (B, K, float32)    1=real, 0=pad

Auxiliary targets:
  y_kill_diff: (B, 1)
  y_gold_diff: (B, 1)
  y_obj_diff:  (B, 1)

Teacher logits:
  lgbm_logit:  (B, 1)             LightGBM baseline prediction
```

### Scaling

StandardScaler applied to numeric features. Spatial coordinates (`x_norm`, `y_norm`, `pos_*`, `dist_*`) are **excluded** from scaling to maintain position invariance.

---

## 8. Model Forward

**Entry:** `train/models.py` (model registry) → `train/deep.py` (training harness)

### 28 Registered Architectures

| Category | Models |
|----------|--------|
| **Baseline** | `lgbm` |
| **RNN** | `rnn_ugru`, `rnn_bigru`, `rnn_ulstm`, `rnn_bilstm`, `rnn_transformer`, `rnn_tcn`, `rnn_mamba` |
| **Hybrid h0** | `hybrid_bigru`, `hybrid_bilstm`, `hybrid_ugru` |
| **GNN** | `gnn_gcn`, `gnn_graphsage`, `gnn_graphtransformer`, `gnn_gatv2`, `gnn_mpnn` |
| **Spatio-Temporal** | `gnn_stgnn`, `gnn_stgcn`, `edge_stgnn`, `ms_stgcn`, `ms_stgnn`, `stgnn_mamba`, `event_xattn` |
| **Fusion** | `fusion_gated_gnn_bigru` |

### GNN Example (GATv2)

```
Input:  node_seq (B, L, 10, F_node) + extra_seq (B, L, D_extra)
                    |
        NodeFeatureAdapter
          ├─ champion_id → Embedding(d)
          ├─ runes → Embedding(d)
          └─ numeric → Linear(d_hidden)
                    |
                concat → (B, L, 10, d_hidden)
                    |
        For each time step t:
          ├─ Build adjacency A from XY positions
          ├─ Role-aware weighting: A_role(i,j) = A_dist(i,j) × R[role(i), role(j)]
          └─ GATv2 message passing: h_t = σ(A · h_t)
                    |
        Temporal attention pooling across L steps
                    |
        Classification head → logit ∈ ℝ
                    |
        sigmoid(logit) → prob ∈ [0, 1]
```

### Loss Function

```
L_total = BCE(logit, y) + λ_k × MSE(pred_kill, y_kill_diff)
                        + λ_g × MSE(pred_gold, y_gold_diff)
                        + λ_o × MSE(pred_obj, y_obj_diff)
```

Focal loss variant available for imbalanced fight outcomes.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RNN_HIDDEN` | 128 | RNN hidden dimension |
| `RNN_LAYERS` | 2 | RNN depth |
| `GNN_DIM` | 96 | GNN hidden dimension |
| `HEAD_HIDDEN` | 128 | Classification head dimension |
| `HEAD_LAYERS` | 2 | Classification head depth |
| `DROPOUT` | 0.20 | Dropout rate |

---

## 9. Evaluation

Predictions are aligned by `ref_key` (not batch position) to handle shuffling, dropped samples, and multi-worker loading.

```python
for batch in dataloader:
    ref_keys = batch["ref_key"]       # List[str]
    logits   = model(batch)           # (B, 1)
    for key, logit in zip(ref_keys, logits):
        logit_map[key] = sigmoid(logit)
```

### Metrics

| Metric | Description |
|--------|-------------|
| AUC | Area Under ROC Curve |
| AP | Average Precision |
| Accuracy | Classification accuracy at threshold 0.5 |
| Precision / Recall / F1 | Per-class and macro |
| Brier Score | Calibration quality |

### Subgroup Analysis

- **By minute**: early / mid / late game fights
- **By gold state**: close / moderate / stomp
- **By patch**: per-patch performance tracking
- **Bootstrap CI**: 5-seed runs `(7, 42, 123, 256, 512)` for confidence intervals

---

## Interpretation

```
prob > 0.5 → model predicts blue team wins the fight
prob < 0.5 → model predicts red team wins the fight
prob ≈ 0.5 → uncertain / close fight
```

The prediction is made at `engage_ts - prediction_gap_ms` (default: at the moment of engagement). The model sees the **game state leading up to the fight** but never the fight outcome itself.

---

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Millisecond anchoring** | All timestamps in ms; sub-minute fight precision |
| **No future leakage** | Global features from strictly-before snapshots; label window starts at engage_ts |
| **Ref-key alignment** | Predictions matched by `match_id\|t_start_ts=<ms>`, not by batch position |
| **Interpolation guards** | XY positions use jump detection + midstep snapping for teleports/deaths |
| **Match-grouped splits** | All fights from one match stay in the same split partition |
| **Patch stratification** | Each split has proportional representation of game patches |
