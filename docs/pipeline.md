# Pipeline: Detect Teamfight & Predict Winner

End-to-end data flow from Riot API timeline JSON to teamfight winner prediction.

```
  Riot API Timeline JSON
          │
          ▼
  ┌─────────────────────┐
  │  [1] Cache Build     │  node_minute [T,10,F], events[], minute_ts[]
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  [2] Fight Detection │  teamfight_v2: kills → clusters → radii → validation
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  [3] FightRef Index  │  "KR_712345|t_start_ts=532000" + train/val/test split
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  [4] Sample Build    │  12 bins × 5s → snapshot node/glob + bin events
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  [5] Label Compute   │  y ∈ {0,1} from kill_diff + alive_diff in label window
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  [6] Model Forward   │  GNN / RNN / Transformer → logit → P(blue wins)
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  [7] Evaluation      │  AUC, AP, Brier (ref_key aligned, 5-seed bootstrap)
  └─────────────────────┘
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
| `xy_raw_minute` | `[T, 10, 2]` | 60s | Raw player positions (X, Y) |
| `minute_ts` | `[T]` | 60s | Frame timestamps in ms |
| `events` | `List[dict]` | ms | All raw game events |

### Node Features (per player, per frame)

```
Position:      x_norm, y_norm           (zeroed for model input — used only for spatial checks)
Resources:     level_norm, xp_norm, curGold_norm, totalGold_norm, gps_norm
CS:            laneCS_norm, jgCS_norm
Status:        alive (0/1), hp_pct, mp_pct
Identity:      champion_id, champion_name_id (categorical → embedding)
Spells:        summoner_spell_1_id, summoner_spell_2_id (categorical)
Runes:         primary_rune_1-4, sub_rune_1-2, style_ids (categorical)
Buffs:         has_baron, has_elder, has_red, has_blue (0/1)
Buff Duration: baron_remain_norm, elder_remain_norm, red_remain_norm, blue_remain_norm
Cooldowns:     ult_level_norm, flash_ready, flash_remain_norm
Vision:        vision_ally_ward_cnt_norm, vision_ward_kill_recent_norm, vision_nearby_score_norm
Dragon Soul:   soul_infernal, soul_ocean, soul_mountain, soul_cloud, soul_hextech, soul_chemtech
Champion Stats: cs_armor, cs_attackDamage, cs_abilityPower, cs_magicResist, ... (25 features)
Damage Stats:  ds_totalDamageDoneToChampions, ds_physicalDamageTaken, ... (12 features)
```

**Total:** `F_node` = 87 features per player per timestep

### Global Features (per frame)

```
Time:          time_norm ∈ [0, 1]
Bans:          blue_ban_0..4, red_ban_0..4 (champion IDs)
Differentials: goldDiff, xpDiff, avgLevelDiff, csDiff_total, csJgDiff, aliveDiff
Cumulative:    killDiff_cum, towerDiff_cum, inhibDiff_cum
               dragonDiff_cum, baronDiff_cum, heraldDiff_cum
               atakhanDiff_cum, plateDiff_cum, hordeDiff_cum
```

---

## 2. Fight Detection (teamfight_v2)

**Entry:** `gameplay/fights.py::detect_fights()` → `detect_fights_teamfight_v2()`

Only kill events create fights. No ward/objective hard gates, no multi-stage guards.
Single definition: **kills → temporal clustering → spatial radii → validation**.

### 2.1 Step-by-Step Algorithm

```
Step 1 ─ Build 5-Second Position Grid
Step 2 ─ Cluster Kills by Time Gap
Step 3 ─ Validate Each Cluster as Teamfight
Step 4 ─ Collect Interactions
Step 5 ─ Post-Fight Outcome (45s window)
Step 6 ─ Classify, Score, Output
```

---

### Step 1: Build 5-Second Position Grid

**Function:** `_build_5s_position_grid()`

Riot API gives player XY at 60-second intervals. We need finer resolution for spatial
checks, so we interpolate to a **5-second dense grid** using two layers:

```
Riot API: 60s frames                  Dense 5s grid
  ┌─ 0:00 ─┬─ 1:00 ─┬─ 2:00 ─┐        ┌─ 0:00 ─ 0:05 ─ 0:10 ─ ... ─ 0:55 ─ 1:00 ─ 1:05 ─ ...
  │ (x,y)  │ (x,y)  │ (x,y)  │  ───►   │  interpolated XY at every 5-second mark
  └────────┴────────┴────────┘        └─ for all 10 players
```

**Layer 1 — Baseline XY Interpolation:**

For each 5-second tick `t` between frame `F_i` (at `ts_i`) and `F_{i+1}` (at `ts_{i+1}`),
the 5s grid uses the XY interpolation curve configured in the codebase
(`cfg.INTERP_XY_CURVE`, default: `exponential` with k=3).
Apply the same curve consistently for baseline and override layers.

```
α_raw = (t - ts_i) / (ts_{i+1} - ts_i)      # α ∈ [0, 1]
α = remap_alpha(α_raw, curve=cfg.INTERP_XY_CURVE)
XY(player, t) = (1 - α) · XY(F_i) + α · XY(F_{i+1})
```

> **Note:** The 5s grid currently uses linear interpolation internally
> (`_build_5s_position_grid`), while `pipeline.py` uses the configured
> curve (exponential by default) with a discontinuity guard for model XY.
> Since model XY is zeroed (`ZERO_XY_NODE_FEATURES = True`), the curve
> choice only affects fight detection spatial checks.

**Layer 2 — Pre-Kill Override:**

For each kill event (processed chronologically), override kill participants' positions:

```
Kill at ts=482000ms at position (8200, 4100)
  Participants: killer(pid=3), victim(pid=7), assists(pid=1, pid=4)

  For each participant:
    prior_frame = last 60s frame before kill
    override interval = [prior_frame_ts ... kill_ts]

    α_kill = (t - prior_frame_ts) / (kill_ts - prior_frame_ts)
    XY(participant, t) = (1-α_kill) · XY(prior_frame) + α_kill · kill_position
```

```
                  prior_frame                       kill event
                      │                                 │
  60s frame ─────────●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●─────── 60s frame
                      │  participant XY overridden to  │
                      │  interpolate toward kill (x,y) │
                      │                                │
                      │  ← override interval →         │
                      │  non-participants: baseline     │
```

Later kills overwrite earlier overrides. This grid is used **ONLY for spatial checks** —
never as model input features.

---

### Step 2: Cluster Kills by Temporal Proximity

**Function:** `_cluster_kills_temporal(gap_ms=18000)`

Kills are sorted by timestamp. Consecutive kills within `18 seconds` remain in the same cluster.
When the next kill exceeds the gap, a new cluster starts.

```
Timeline (ms):
  K1        K2    K3              K4   K5      K6
  │         │     │               │    │       │
  120000    125000 131000          180000 185000 192000
  │←──5s──→│←─6s─→│               │←5s→│←─7s──→│
  │    within 18s gap              │   within 18s gap
  │                                │
  └───── Cluster A ────────┘      └──── Cluster B ─────┘
    first_kill: 120000               first_kill: 180000
    last_kill:  131000               last_kill:  192000
    center: K1 position              center: K4 position
                    ▲ 49s gap ▲
                   (> 18s → split)
```

Each cluster produces:
- `first_kill_ts`, `last_kill_ts`
- `fight_center` = first kill's (x, y) position
- `participants` = set of all killer/victim/assist IDs
- `n_kills` = number of kills in cluster

---

### Step 3: Validate Each Cluster as Teamfight

For each kill cluster, determine if it qualifies as a teamfight:

```
3a. Compute engage time
    engage_ts = first_kill_ts − 10,000ms
    (clamped to game start)

3b. Check context bounds
    engage_ts must be at least 60s into the game
    engage_ts + horizon must not exceed game end

3c. Check alive count
    At engage_ts: both teams must have ≥ 2 alive champions

3d. SPATIAL VALIDATION (§4A): Radius 1800 check
    At engage_ts, look up all 10 positions from the 5s grid.
    Count how many are within radius=1800 of fight_center.
    Require: ≥ 2 blue AND ≥ 2 red within radius.
```

```
          Radius 1800 check at engage_ts
          ┌─────────────────────────────────┐
          │                                 │
          │     ●B1   ●B2                   │
          │              ⊕ fight_center     │
          │     ●R1   ●R2                   │
          │                                 │   ● = player inside
          │                                 │   ○ = player outside
          └─────────────────────────────────┘
                                  ○B3 ○R3 ○B4 ○R4 ○B5 ○R5

    blue_in_radius = 2 (B1, B2)  ≥ 2 ✓
    red_in_radius  = 2 (R1, R2)  ≥ 2 ✓
    → VALID TEAMFIGHT
```

If validation fails (e.g., 1v1 pick), the cluster is rejected:

```
          ●B1              ⊕ center
                           ●R1
    blue_in = 1  < 2 ✗
    → REJECTED (pick, not teamfight)
```

---

### Step 4: Collect Interactions (Radius 3000)

**Function:** `_collect_interactions_in_radius()`

Non-kill events during `[engage_ts, last_kill_ts]` within **radius 3000** of fight center
are counted as fight interactions.

```
          ┌──────── Radius 3000 ────────┐
          │                             │
          │   ┌── Radius 1800 ──┐       │
          │   │                 │       │
          │   │  ⊕ center       │       │
          │   │  (validity)     │       │
          │   └─────────────────┘       │
          │   (interactions counted)    │
          └─────────────────────────────┘
```

**Important:** Objective and tower events (ELITE_MONSTER_KILL, BUILDING_KILL, TURRET_PLATE_DESTROYED)
are **NOT** counted as radius-3000 interactions. They are tracked only in the
post-fight outcome window (Step 5), which is radius-independent. This prevents
double-counting the same event in both interactions and outcome.

Only position-based events (wards, summoner spells, etc.) are collected here.

---

### Step 5: Post-Fight Outcome Window (45 seconds)

**Function:** `_compute_postfight_outcome()`

After the last kill in the cluster, a **45-second window** captures the consequences:

```
  ─── fight ───                    ─── post-fight window (45s) ───
  [engage ... last_kill]           [last_kill ... last_kill + 45000ms]
                    │                              │
                    │  Collect:                     │
                    │    • objectives taken         │
                    │    • towers destroyed         │
                    │    • gold differential        │
                    │                              │
```

This captures whether the winning team converted kills into map objectives.

---

### Step 6: Classify, Score, Output

Each validated fight is classified by spatial location and enriched with outcome data.

**Fight Type Classification** (`classify_fight_type()`):

```
  Is fight center near Baron pit (< 1500)?     → objective_baron
  Is fight center near Dragon pit (< 1500)?    → objective_dragon
  Is fight center near Rift Herald (< 1500)?   → objective_riftherald
  Is fight center near a tower (< 1000)?       → tower_dive
  Is fight center near a base (< 3000)?        → base_fight
  Are there ≥ 8 proximity pairs?               → teamfight
  Are there ≥ 4 proximity pairs?               → skirmish
  Otherwise                                    → pick
```

**Fight Outcome** (`compute_fight_outcome()`):

Counts kills, deaths, assists, gold swing, towers, objectives in the label window
`[engage_ts, horizon_end_ts)` — blue team kills minus red team kills determines the winner.

### Complete Timeline Diagram

```
  Game Timeline (ms)
  ════════════════════════════════════════════════════════════════════

  0        60000      120000     180000     240000     300000
  ├──────────┼──────────┼──────────┼──────────┼──────────┤
  │          │          │          │          │          │
  │  60s frame snapshots from Riot API (node_minute)    │
  │                                                     │
  │                    Kill K1 ────┐                     │
  │                    Kill K2 ──┐ │ < 18s gap           │
  │                    Kill K3 ┐ │ │  (same cluster)     │
  │                            │ │ │                     │
  │                            ▼ ▼ ▼                     │
  │                     ┌──────────────┐                 │
  │                     │ Kill Cluster │                 │
  │                     │ first: K1    │                 │
  │                     │ last:  K3    │                 │
  │                     │ center: K1xy │                 │
  │                     └──────┬───────┘                 │
  │                            │                         │
  │              ┌─────────────┤                         │
  │              ▼             ▼                         │
  │         engage_ts    first_kill_ts                   │
  │         (K1 - 10s)   (K1)                           │
  │              │                                       │
  │    ┌─────────┤                                       │
  │    │ Radius  │                                       │
  │    │ 1800    │ → ≥2 blue + ≥2 red? → VALID         │
  │    │ check   │                                       │
  │    └─────────┤                                       │
  │              │                                       │
  │              │◄─── fight time window ──►│            │
  │              │   [engage ... last_kill]  │            │
  │              │      │                   │            │
  │              │      │ Radius 3000       │            │
  │              │      │ interactions      │            │
  │              │                          │            │
  │              │                     last_kill_ts      │
  │              │                          │            │
  │              │                          │◄── 45s ──►│
  │              │                          │ post-fight │
  │              │                          │ objectives │
  │              │                          │ gold swing │
  │                                                     │
  ════════════════════════════════════════════════════════════════════
```

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TF2_KILL_CLUSTER_GAP_MS` | 18,000 | Max gap between kills in same cluster |
| `TF2_ENGAGE_PRE_KILL_MS` | 10,000 | How far before first kill = engage time |
| `TF2_VALIDITY_RADIUS` | 1,800 | Radius for teamfight validation (≥2 per team) |
| `TF2_INTERACTION_RADIUS` | 3,000 | Radius for counting fight interactions |
| `TF2_POST_FIGHT_WINDOW_MS` | 45,000 | Post-fight outcome window |
| `TF2_MIN_PER_TEAM` | 2 | Minimum champions per team in validity radius |
| `FIGHT_MIN_GAP_MS` | 60,000 | Minimum spacing between detected fights |
| `MAX_MERGED_FIGHT_DURATION_MS` | 120,000 | Maximum fight duration cap |

### Detection Output

```python
{
    "engage_ts":        int,    # ms — fight start (primary anchor)
    "horizon_end_ts":   int,    # ms — label window end
    "first_kill_ts":    int,    # ms — first kill in cluster
    "last_kill_ts":     int,    # ms — last kill in cluster
    "centroid_x":       float,  # fight center X (from first kill)
    "centroid_y":       float,  # fight center Y (from first kill)
    "fight_type":       str,    # teamfight / skirmish / objective_baron / tower_dive / ...
    "outcome":          dict,   # kills, deaths, gold, towers per team
    "post_fight_outcome": dict, # 45s window: objectives, towers, gold swing
}
```

---

## 3. FightRef Index

**Entry:** `data/index_split.py::build_fight_index()`

Each detected fight becomes a `FightRef` — the unique identifier that tracks a sample through
the entire pipeline.

```python
FightRef(
    match_id    = "KR_7123456789",
    patch       = "14.10",
    t_start     = 8,            # minute index
    t_start_ts  = 532000,       # engage timestamp in ms (primary key)
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

All fights from one match stay in the same split (no match leakage).

---

## 4. Sample Build (Observation Window)

**Entry:** `gameplay/pipeline.py::build_ms_sequence()`

Given a `FightRef` with `engage_ts`, builds the **observation window** (what the model sees)
and the **label window** (what we predict).

### Timeline Layout

```
  ◄──────────── observation window (60s) ──────────────►
  │                                                     │
  │  bin0   bin1   bin2   ...   bin10  bin11             │
  │ [0-5s] [5-10s] [10-15s]         [50-55s] [55-60s]  │
  │                                                     │
  start_ms                                           end_ms = engage_ts
  (engage_ts − 60s)                                     │
                                                        │
                                        ◄─── label window (60s) ───►
                                        │                           │
                                    engage_ts               label_end_ts
                                   (= fight start)      (= engage + horizon)
                                        │                           │
                                        │   kills, deaths, gold     │
                                        │   counted here → label    │
```

**Key principle:** The model sees 60 seconds of game state **before the fight starts**.
It never sees the fight outcome — that's the label.

### Time Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ctx_ms` | 60,000 | Observation window duration |
| `bin_ms` | 5,000 | Time bin size |
| `horizon_ms` | 60,000 | Label window duration |
| `prediction_gap_ms` | 0 | Gap between observation end and engage_ts |

**Derived:** `L = ctx_ms / bin_ms = 60000 / 5000 = 12 time steps`

### Per-Bin Computation

For each of the 12 bins, the midpoint timestamp `q` is computed:

```
  bin_i:  b0 = start_ms + i × 5000
          b1 = start_ms + (i+1) × 5000
          q  = b0 + 2500  (midpoint)
```

At each midpoint `q`:

```
  node_i = snapshot from nearest 60s frame STRICTLY BEFORE q
      → Piecewise-constant (step-hold / ffill): NO interpolation of
        scalar features (stats, items, buffs, etc.)
      → XY positions zeroed (ZERO_XY_NODE_FEATURES = True)
      → Same snapshot repeated for all bins within one 60s frame interval
      → Snapshot changes only when bin midpoint crosses a frame boundary
      → Shape: (10, F_node)

  glob_i = snapshot from nearest 60s frame STRICTLY BEFORE q
      → Same piecewise-constant rule — no interpolation, no future leakage
      → Shape: (F_global,)

  ev_i = aggregate_events(cache, team_map, b0, b1)
      → Count kills, spells, objectives, wards in [b0, b1)
      → Shape: (F_event,)

  item_i = aggregate_items(cache, team_map, b0, b1)
      → Hash item purchases within [b0, b1)
      → Shape: (F_item,)
```

> **Rule: "XY만 보간, 나머지 피처는 보간 금지"**
> Node scalars and global features use strict-before 60s snapshots
> (piecewise-constant). Only XY positions are interpolated (for the 5s
> position grid in fight detection), and even those are zeroed in model input.

### Feature Handling Rules

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  RULE: "XY만 보간, 나머지 피처는 보간 금지 / 모델 입력은 스냅샷"      │
  │                                                                  │
  │  1. Fight Detection (5s grid):                                   │
  │     XY IS interpolated — dense 5s grid for radius checks         │
  │     This is internal to detect_fights_teamfight_v2()             │
  │                                                                  │
  │  2. Model Input — Node/Global (observation window):              │
  │     ✗ NO interpolation of scalar features                        │
  │     Use strict-before 60s snapshot (piecewise-constant / ffill)  │
  │     → INTERP_SCALARS_METHOD = "ffill"                            │
  │                                                                  │
  │  3. Model Input — XY:                                            │
  │     XY is ZEROED — x_norm=0, y_norm=0 in every bin              │
  │     Model predicts fight outcome from stats, not XY              │
  │     Prevents model from memorizing map-position bias             │
  │                                                                  │
  │  4. Events/Items:                                                │
  │     Bin-level aggregation [b0, b1) — this is NOT interpolation   │
  │     It counts raw events that occurred in the bin time interval   │
  └──────────────────────────────────────────────────────────────────┘
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

### Full Sample Build Diagram

```
  Game:  0:00    1:00    2:00    3:00    4:00    5:00    6:00    7:00    8:00
         │       │       │       │       │       │       │       │       │
         ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
                 60s frame snapshots from Riot API

  Detected fight: engage_ts = 420000 (7:00)

  Observation window: [360000, 420000] = 6:00 → 7:00

       6:00                                                         7:00
        │  bin0  │  bin1  │  bin2  │  ...  │ bin10  │ bin11  │       │
        │ 360-365│ 365-370│ 370-375│       │ 410-415│ 415-420│       │
        │  q=362 │  q=367 │  q=372 │       │  q=412 │  q=417 │       │
        │        │        │        │       │        │        │       │
        ▼        ▼        ▼        ▼       ▼        ▼        ▼       │
    snap      snap      snap              snap     snap              │
    node+glob node+glob node+glob         node+glob node+glob       │
    + events  + events  + events          + events  + events         │
                                                                     │
                                                               engage_ts
                                                                     │
                                                                     ▼
                                                            Label window
                                                            [420000, 480000]
                                                             7:00 → 8:00
                                                            count kills →
                                                            compute y
```

---

## 5. Label Computation

**Entry:** `gameplay/pipeline.py::compute_label_targets()`

### Definition of "Winning a Teamfight"

Two aspects define the fight outcome:

| Aspect | Window | What It Captures |
|--------|--------|------------------|
| **Fight Result** (primary label) | `[engage_ts, engage_ts + horizon)` | Who won the fight itself (kills, survival) |
| **Post-Fight Conversion** (auxiliary) | `[last_kill_ts, last_kill_ts + 45s]` | What the winner gained (towers, objectives, gold) |

The **primary binary label** uses fight-result signals (kill_diff + alive_diff).
Post-fight conversion metrics (gold, towers, objectives) are tracked as
**auxiliary targets for multi-task learning** and **evaluation metrics**.

### Primary Label: `kill_survival` (default)

Events within the label window `[engage_ts, engage_ts + horizon_ms)` determine the fight outcome.

```
  Score = W_KILL × (blue_kills − red_kills) + W_ALIVE × (blue_alive − red_alive)

  W_KILL  = 1.0   (kill differential weight)
  W_ALIVE = 0.3   (alive-at-end differential weight)

  y = 1  if Score > 0   → blue team wins the fight
  y = 0  if Score < 0   → red team wins the fight
  tie   → dropped (LABEL_TIE_STRATEGY = "random" assigns randomly)
```

**Example:**

```
  Label window [420000, 480000]:
    Blue kills: 3, Red kills: 1  →  kill_diff = +2
    Blue alive at end: 4, Red alive at end: 2  →  alive_diff = +2

    Score = 1.0 × 2 + 0.3 × 2 = 2.6 > 0  →  y = 1 (blue wins)
```

### Auxiliary Targets (multi-task learning + evaluation)

Post-fight conversion signals from the 45s outcome window are used as
auxiliary regression targets and evaluation metrics. They capture whether
the fight winner successfully *converted* kills into map advantages.

| Target | Source | Normalization | Description |
|--------|--------|---------------|-------------|
| `y_kill_diff` | fight window | kill_diff / 5.0 | Normalized kill differential |
| `y_gold_diff` | fight window | gold_diff / 1000.0 | Normalized gold swing |
| `y_obj_diff` | fight window | obj_diff / 5.0 | Normalized objective differential |
| `y_alive_diff_raw` | fight window | raw count | Alive count differential |
| `post_gold_diff` | 45s outcome | raw gold | Gold swing after fight |
| `post_tower_diff` | 45s outcome | raw count | Towers taken after fight |
| `post_obj_diff` | 45s outcome | raw count | Objectives taken after fight |

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Primary label (y):   fight window kill/alive signals       │
  │  Auxiliary targets:   fight window gold/obj + 45s conversion│
  │                                                             │
  │  The model predicts "who wins the fight" (primary y).       │
  │  Auxiliary targets help the model learn richer signals      │
  │  about fight consequences (multi-task loss).                │
  └─────────────────────────────────────────────────────────────┘
```

---

## 6. Tensor Collation & Model Forward

**Entry:** `data/dataset.py::InMemoryFightDataset` → `train/deep.py`

### Tensor Layout (GNN / RNN)

```
  node_seq:   (B, 12, 10, F_node)    float32   per-player temporal sequence
  extra_seq:  (B, 12, D_extra)       float32   macro + spatial features
  y:          (B, 1)                 float32   binary label

  Optional:
    event_type:  (B, K)   int64    event type hash
    event_actor: (B, K)   int64    participant ID
    event_cont:  (B, K, 5) float32  [t_rel, dt_end, x, y, val]
    event_mask:  (B, K)   float32  1=real, 0=pad
```

### Model Architecture (GATv2 Example)

```
  Input
    │
    ▼
  node_seq (B, 12, 10, F_node)
    │
    ├──► champion_id ──► Embedding(d)  ─┐
    ├──► rune_ids    ──► Embedding(d)  ─┤
    └──► numeric     ──► Linear(d)     ─┤
                                        ▼
                                   concat → (B, 12, 10, d_hidden)
                                        │
                        ┌───────────────┤ For each time step t:
                        │               │
                        │   Build adjacency A from XY positions
                        │   (soft Gaussian: A_ij = exp(-d²/2σ²))
                        │               │
                        │   GATv2 multi-head attention
                        │   h_t = σ(α_ij · W · h_j)
                        │               │
                        └───────────────┤
                                        ▼
                              (B, 12, 10, d_hidden)
                                        │
                              Temporal attention pooling
                              across 12 time steps
                                        │
                                        ▼
                              (B, d_pool)
                                        │
                              Classification head
                              Linear → ReLU → Dropout → Linear
                                        │
                                        ▼
                              logit ∈ ℝ
                                        │
                              sigmoid(logit)
                                        │
                                        ▼
                              P(blue wins) ∈ [0, 1]
```

### Loss Function

```
  L_total = BCE(logit, y)
          + λ_kill × MSE(pred_kill, y_kill_diff)
          + λ_gold × MSE(pred_gold, y_gold_diff)
          + λ_obj  × MSE(pred_obj,  y_obj_diff)
```

### Registered Architectures (28 models)

| Category | Models |
|----------|--------|
| **Baseline** | `lgbm` (LightGBM) |
| **RNN** | `rnn_bigru`, `rnn_bilstm`, `rnn_transformer`, `rnn_tcn`, `rnn_mamba` |
| **Hybrid h0** | `hybrid_bigru`, `hybrid_bilstm` |
| **GNN** | `gnn_graphsage`, `gnn_gatv2`, `gnn_mpnn` |
| **Spatio-Temporal** | `gnn_stgnn`, `stgcn`, `stgnn_mamba`, `event_xattn`, `ms_dyngraph` |
| **Fusion** | `fusion_gated_gnn_bigru` |

---

## 7. Evaluation

Predictions are aligned by `ref_key` (not batch position) to handle shuffling,
dropped samples, and multi-worker loading.

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
- **By gold state**: close (< 2000) / moderate / stomp (> 5000)
- **By patch**: per-patch performance tracking
- **Bootstrap CI**: 5-seed runs `(7, 42, 123, 256, 512)` for confidence intervals

---

## Full End-to-End Example

```
  MATCH: KR_7123456789, Patch 14.10, Duration 32:00
  ════════════════════════════════════════════════════════

  [1] Cache Build
      → 33 minute frames (0:00 → 32:00)
      → node_minute: [33, 10, 87]
      → 847 raw events (kills, objectives, wards, ...)

  [2] Fight Detection (teamfight_v2)
      → Extract 28 CHAMPION_KILL events
      → Cluster temporally (gap=18s): 6 clusters
      → Build 5s position grid: [385 timesteps, 10, 2]
      → Validate each cluster:
          Cluster 1 (3 kills, 7:10-7:25): 3v3 at radius 1800 → ✓ teamfight
          Cluster 2 (1 kill, 11:40):      1v1 at radius 1800 → ✗ rejected (pick)
          Cluster 3 (4 kills, 15:20-15:45): 4v4 at radius 1800 → ✓ teamfight
          Cluster 4 (2 kills, 20:05-20:12): 2v3 at radius 1800 → ✓ teamfight
          Cluster 5 (5 kills, 26:30-27:00): 5v5 at radius 1800 → ✓ teamfight
          Cluster 6 (2 kills, 31:10-31:15): horizon exceeds game → ✗ rejected
      → 4 fights detected

  [3] FightRef Index
      → Fight 1: "KR_7123456789|t_start_ts=420000"
      → Fight 3: "KR_7123456789|t_start_ts=910000"
      → Fight 4: "KR_7123456789|t_start_ts=1195000"
      → Fight 5: "KR_7123456789|t_start_ts=1580000"
      → Split: match grouped into "train" partition

  [4] Sample Build (Fight 1: engage_ts = 420000)
      → Observation window: [360000, 420000] (6:00 → 7:00)
      → 12 bins × 5s each
      → Per bin: snapshot node+global (strict-before 60s frame, no interpolation),
                 aggregate events, hash items
      → XY zeroed in all bins
      → node_seq: [12, 10, 87], glob_seq: [12, 27], ev_seq: [12, 48]

  [5] Label (Fight 1)
      → Label window: [420000, 480000] (7:00 → 8:00)
      → Events in window: 3 blue kills, 1 red kill
      → Blue alive at 8:00: 4, Red alive: 2
      → Score = 1.0 × 2 + 0.3 × 2 = 2.6 → y = 1 (blue wins)

  [6] Model Prediction
      → GATv2 forward: node_seq + extra → logit = 1.34
      → P(blue wins) = sigmoid(1.34) = 0.79
      → Prediction: blue team wins (confidence 79%)

  [7] Evaluation
      → Aligned by ref_key: "KR_7123456789|t_start_ts=420000"
      → True label: y=1, Predicted: 0.79 → correct
      → Contributes to AUC, AP, accuracy metrics
```

---

## Interpretation

```
  P(blue wins) > 0.5  →  model predicts blue team wins the fight
  P(blue wins) < 0.5  →  model predicts red team wins the fight
  P(blue wins) ≈ 0.5  →  uncertain / close fight
```

The prediction is made at `engage_ts` (the moment of engagement). The model sees
the **game state leading up to the fight** (60s observation window) but never the
fight outcome itself.

---

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Kills-only creation** | Only CHAMPION_KILL events create fights — no ward/objective triggers |
| **XY만 보간, 나머지 금지** | Node/global use strict-before 60s snapshot (ffill); only XY is interpolated (for spatial checks) |
| **No future leakage** | Node + global features from strictly-before snapshots; label window starts at engage_ts |
| **XY excluded from model** | Positions used only for spatial detection, zeroed in model input |
| **No double-counting** | Objectives/towers tracked only in post-fight outcome (Step 5), not in radius-3000 interactions (Step 4) |
| **Millisecond anchoring** | All timestamps in ms; sub-minute fight precision via 5s grid |
| **Ref-key alignment** | Predictions matched by `match_id\|t_start_ts=<ms>`, not by batch position |
| **Match-grouped splits** | All fights from one match stay in the same split partition |
| **Patch stratification** | Each split has proportional representation of game patches |
| **Post-fight conversion** | 45-second window captures objective/tower/gold conversion after fight |
