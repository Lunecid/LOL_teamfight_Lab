# Feature Engineering Specification

Complete definition of all feature sets, dimensions, normalization rules, and constant/quasi-constant classification used in the LOL Teamfight Lab pipeline.

---

## Feature Dimension Summary

| Feature Group | Symbol | Dimensions | Scope |
|---------------|--------|------------|-------|
| Node features | F_node | **76** per player per timestep | Per-player |
| Global features | F_global | **26** per timestep | Team-level |
| Event features | F_event | **44** per timestep per bin | Per-bin aggregate |
| Temporal bins | L | **6** (30s / 5s) | Per-sample |
| Players | N | **10** (5 per team) | Per-sample |
| Champion stats | F_cs | 25 | Subset of F_node |
| Damage stats | F_ds | 12 | Subset of F_node |
| Rune features | F_rune | 11 | Subset of F_node |
| Ban features | F_ban | 10 | Subset of F_global |
| Event tokens (max) | K | 64 | For cross-attention models |
| Event continuous dim | D_e | 12 | Per event token |

### Tensor Shapes at Model Input

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `node_seq` | `(B, 6, 10, 76)` | float32 | Per-player temporal sequence |
| `extra_seq` | `(B, 6, D_extra)` | float32 | Flattened macro + spatial features |
| `y` | `(B, 1)` | float32 | Binary label |
| `event_type` | `(B, K)` | int64 | Event type hash (optional) |
| `event_actor` | `(B, K)` | int64 | Participant ID (optional) |
| `event_cont` | `(B, K, 5)` | float32 | [t_rel, dt_end, x, y, val] (optional) |
| `event_mask` | `(B, K)` | float32 | 1=real event, 0=padding (optional) |

---

## 1. Node Features (F_node = 76)

Per-player features extracted at each 60-second frame. Defined in `core/config.py::NODE_FEATURE_NAMES`.

### 1.1 Snapshot Features (17 features)

Defined in `NODE_SNAPSHOT_FEATURE_NAMES`:

| # | Feature | Type | Range | Description |
|---|---------|------|-------|-------------|
| 0 | `champion_id` | Categorical | Integer ID | Riot champion numeric ID (-> embedding) |
| 1 | `champion_name_id` | Categorical | Integer hash | Champion name hash (-> embedding) |
| 2 | `summoner_spell_1_id` | Categorical | Integer ID | First summoner spell (-> embedding) |
| 3 | `summoner_spell_2_id` | Categorical | Integer ID | Second summoner spell (-> embedding) |
| 4 | `x_norm` | Continuous | [0, 1] | Normalized X position (x / MAP_MAX, **zeroed in model**) |
| 5 | `y_norm` | Continuous | [0, 1] | Normalized Y position (y / MAP_MAX, **zeroed in model**) |
| 6 | `level_norm` | Continuous | [0, 1] | Champion level / 18 |
| 7 | `xp_norm` | Continuous | [0, ~1] | Experience normalized |
| 8 | `curGold_norm` | Continuous | [0, ~1] | Current (unspent) gold normalized |
| 9 | `totalGold_norm` | Continuous | [0, ~1] | Total earned gold normalized |
| 10 | `gps_norm` | Continuous | [0, ~1] | Gold per second normalized |
| 11 | `laneCS_norm` | Continuous | [0, ~1] | Lane creep score normalized |
| 12 | `jgCS_norm` | Continuous | [0, ~1] | Jungle creep score normalized |
| 13 | `ccTime_norm` | Continuous | [0, ~1] | Crowd-control time dealt (normalized) |
| 14 | `hp_pct` | Continuous | [0, 1] | Current HP / Max HP |
| 15 | `mp_pct` | Continuous | [0, 1] | Current MP / Max MP (0 for manaless) |
| 16 | `alive` | Binary | {0, 1} | 1 if champion is alive |

### 1.2 Status Features (11 features)

Defined in `NODE_STATUS_FEATURE_NAMES`:

| # | Feature | Type | Range | Description |
|---|---------|------|-------|-------------|
| 17 | `has_baron` | Binary | {0, 1} | Team has Baron Nashor buff |
| 18 | `has_elder` | Binary | {0, 1} | Team has Elder Dragon buff |
| 19 | `baron_remain_norm` | Continuous | [0, 1] | Baron buff remaining / 180s |
| 20 | `elder_remain_norm` | Continuous | [0, 1] | Elder buff remaining / 150s |
| 21 | `soul_infernal` | Binary | {0, 1} | Team has Infernal Dragon Soul |
| 22 | `soul_ocean` | Binary | {0, 1} | Team has Ocean Dragon Soul |
| 23 | `soul_mountain` | Binary | {0, 1} | Team has Mountain Dragon Soul |
| 24 | `soul_cloud` | Binary | {0, 1} | Team has Cloud Dragon Soul |
| 25 | `soul_hextech` | Binary | {0, 1} | Team has Hextech Dragon Soul |
| 26 | `soul_chemtech` | Binary | {0, 1} | Team has Chemtech Dragon Soul |
| 27 | `ult_level_norm` | Continuous | [0, 1] | Ultimate ability level / 3 |

### 1.3 Rune Features (11 features)

Defined in `RUNE_FEATURE_NAMES`:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 28 | `primary_style_id` | Categorical | Primary rune tree ID (e.g., Precision, Domination) |
| 29 | `sub_style_id` | Categorical | Secondary rune tree ID |
| 30 | `primary_rune_1` | Categorical | Keystone rune ID |
| 31 | `primary_rune_2` | Categorical | Primary tree row 1 |
| 32 | `primary_rune_3` | Categorical | Primary tree row 2 |
| 33 | `primary_rune_4` | Categorical | Primary tree row 3 |
| 34 | `sub_rune_1` | Categorical | Secondary tree rune 1 |
| 35 | `sub_rune_2` | Categorical | Secondary tree rune 2 |
| 36 | `stat_perk_offense` | Categorical | Offense stat shard |
| 37 | `stat_perk_flex` | Categorical | Flex stat shard |
| 38 | `stat_perk_defense` | Categorical | Defense stat shard |

### 1.4 Champion Stats (25 features)

Defined in `CHAMPION_STATS_KEYS`, prefixed with `cs_`:

| # | Feature | Description |
|---|---------|-------------|
| 39 | `cs_abilityHaste` | Ability haste |
| 40 | `cs_abilityPower` | Ability power |
| 41 | `cs_armor` | Armor |
| 42 | `cs_armorPen` | Flat armor penetration |
| 43 | `cs_armorPenPercent` | % armor penetration |
| 44 | `cs_attackDamage` | Attack damage |
| 45 | `cs_attackSpeed` | Attack speed |
| 46 | `cs_bonusArmorPenPercent` | Bonus armor pen % |
| 47 | `cs_bonusMagicPenPercent` | Bonus magic pen % |
| 48 | `cs_ccReduction` | Tenacity (CC reduction %) |
| 49 | `cs_cooldownReduction` | Cooldown reduction % |
| 50 | `cs_health` | Current health |
| 51 | `cs_healthMax` | Maximum health |
| 52 | `cs_healthRegen` | Health regeneration per 5s |
| 53 | `cs_lifesteal` | Life steal % |
| 54 | `cs_magicPen` | Flat magic penetration |
| 55 | `cs_magicPenPercent` | % magic penetration |
| 56 | `cs_magicResist` | Magic resistance |
| 57 | `cs_movementSpeed` | Movement speed |
| 58 | `cs_omnivamp` | Omnivamp % |
| 59 | `cs_physicalVamp` | Physical vamp % |
| 60 | `cs_power` | Current resource (mana/energy) |
| 61 | `cs_powerMax` | Maximum resource |
| 62 | `cs_powerRegen` | Resource regeneration per 5s |
| 63 | `cs_spellVamp` | Spell vamp % |

**Auto-correction:** Keys in `CHAMPION_STATS_DIV100_KEYS` (attackSpeed, percentages, vamp stats) are auto-corrected by `/100` when `|v| > 2` to handle Riot API inconsistencies across patches/regions.

### 1.5 Damage Stats (12 features)

Defined in `DAMAGE_STATS_KEYS`, prefixed with `ds_`:

| # | Feature | Description |
|---|---------|-------------|
| 64 | `ds_physicalDamageDone` | Physical damage dealt (total) |
| 65 | `ds_magicDamageDone` | Magic damage dealt (total) |
| 66 | `ds_trueDamageDone` | True damage dealt (total) |
| 67 | `ds_totalDamageDone` | Total damage dealt (all types) |
| 68 | `ds_physicalDamageDoneToChampions` | Physical damage to champions |
| 69 | `ds_magicDamageDoneToChampions` | Magic damage to champions |
| 70 | `ds_trueDamageDoneToChampions` | True damage to champions |
| 71 | `ds_totalDamageDoneToChampions` | Total damage to champions |
| 72 | `ds_physicalDamageTaken` | Physical damage taken |
| 73 | `ds_magicDamageTaken` | Magic damage taken |
| 74 | `ds_trueDamageTaken` | True damage taken |
| 75 | `ds_totalDamageTaken` | Total damage taken |

### Total: F_node = 17 + 11 + 11 + 25 + 12 = **76 features**

The exact dimension is computed as `len(NODE_FEATURE_NAMES)` from the canonical list in `core/config.py`.

---

## 2. Global Features (F_global = 26)

Team-level features extracted at each 60-second frame. Defined in `core/config.py::GLOBAL_FEATURE_NAMES`.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 0 | `time_norm` | Continuous [0, 1] | Normalized game time (current / total) |
| 1-5 | `blue_ban_0..4` | Categorical | Blue team champion ban IDs (5 bans) |
| 6-10 | `red_ban_0..4` | Categorical | Red team champion ban IDs (5 bans) |
| 11 | `goldDiff` | Continuous | (Blue total gold - Red total gold), normalized |
| 12 | `xpDiff` | Continuous | (Blue total XP - Red total XP), normalized |
| 13 | `avgLevelDiff` | Continuous | Average level difference (Blue - Red) |
| 14 | `csDiff_total` | Continuous | Total CS difference |
| 15 | `csJgDiff` | Continuous | Jungle CS difference |
| 16 | `aliveDiff` | Continuous | Alive champion count difference |
| 17 | `killDiff_cum` | Continuous | Cumulative kill differential |
| 18 | `towerDiff_cum` | Continuous | Cumulative tower differential |
| 19 | `inhibDiff_cum` | Continuous | Cumulative inhibitor differential |
| 20 | `dragonDiff_cum` | Continuous | Cumulative dragon differential |
| 21 | `baronDiff_cum` | Continuous | Cumulative baron differential |
| 22 | `heraldDiff_cum` | Continuous | Cumulative herald differential |
| 23 | `atakhanDiff_cum` | Continuous | Cumulative atakhan differential |
| 24 | `plateDiff_cum` | Continuous | Cumulative tower plate differential |
| 25 | `hordeDiff_cum` | Continuous | Cumulative void grub (horde) differential |

---

## 3. Event Features (F_event = 44)

Per-bin aggregated event counts within each 5-second time bin `[b0, b1)`. Defined in `core/config.py::EVENT_FEATURE_NAMES`.

All features are split by team: `_t100` (Blue) and `_t200` (Red).

| Feature Pair (Blue / Red) | Description |
|---------------------------|-------------|
| `kills_t100` / `kills_t200` | Champion kills |
| `bounty_t100` / `bounty_t200` | Bounty gold from kills |
| `shutdown_kill_t100` / `shutdown_kill_t200` | Shutdown kills |
| `killstreak_t100` / `killstreak_t200` | Kill streak events |
| `multikill_t100` / `multikill_t200` | Multi-kill events (double, triple, quadra, penta) |
| `ace_t100` / `ace_t200` | Ace events |
| `dragon_t100` / `dragon_t200` | Dragon kills |
| `baron_t100` / `baron_t200` | Baron Nashor kills |
| `herald_t100` / `herald_t200` | Rift Herald kills |
| `atakhan_t100` / `atakhan_t200` | Atakhan kills |
| `horde_t100` / `horde_t200` | Void Grub (horde) kills |
| `tower_t100` / `tower_t200` | Tower destructions |
| `inhib_t100` / `inhib_t200` | Inhibitor destructions |
| `plate_t100` / `plate_t200` | Tower plates destroyed |
| `obj_bounty_t100` / `obj_bounty_t200` | Objective bounties |
| `ward_placed_t100` / `ward_placed_t200` | Wards placed |
| `ward_kill_t100` / `ward_kill_t200` | Wards destroyed |
| `control_ward_placed_t100` / `control_ward_placed_t200` | Control wards placed |
| `control_ward_kill_t100` / `control_ward_kill_t200` | Control wards destroyed |
| `item_pur_t100` / `item_pur_t200` | Item purchases |
| `item_sold_t100` / `item_sold_t200` | Items sold |
| `item_undo_t100` / `item_undo_t200` | Item purchase undos |

**Total:** 22 feature pairs x 2 teams = **44 features per bin**.

---

## 4. Event Tokens (Cross-Attention Models)

For event cross-attention models (`EventXAttnSTModel`), discrete game events are encoded as individual tokens rather than bin aggregates.

| Field | Shape | Description |
|-------|-------|-------------|
| `event_type` | `(K,)` int64 | Event type hash ID |
| `event_actor` | `(K,)` int64 | Participant ID (0-9) |
| `event_cont` | `(K, D_e)` float32 | Continuous features |
| `event_mask` | `(K,)` float32 | 1=real, 0=padding |

Where `K = 64` (max tokens), `D_e = 12`.

Continuous features per event token include:
- `t_rel` -- relative timestamp within observation window
- `dt_end` -- time to engagement
- `x`, `y` -- event position (normalized)
- `val` -- event value (gold, damage, etc.)
- Additional domain features (importance prior, team indicator, etc.)

---

## 5. Feature Normalization

### Node Feature Normalization

| Feature Category | Method | Details |
|-----------------|--------|---------|
| Scalar stats (level, gold, XP, CS, HP, MP) | Z-score | `(v - mu) / sigma`, clipped to [-10, 10] |
| Champion stats (`cs_*`) | Pre-normalized | Deterministic denominators from game knowledge |
| Damage stats (`ds_*`) | log1p | `log(1 + v)` then z-score |
| Categorical IDs (champion, runes, spells) | Preserved as integers | Fed to embedding layers |
| Spatial (x_norm, y_norm) | `v / MAP_MAX` | Then **zeroed** in model input |
| Binary (alive, has_baron, soul_*) | No normalization | Already in {0, 1} |
| Buff duration (baron_remain, elder_remain) | `remaining_sec / total_duration_sec` | See BUFF_DUR_SEC |

### Scaler Exclusions

`SCALER_EXCLUDE_PREFIXES` prevents double-normalization of:
- `cs_` (champion stats) -- already normalized by deterministic denominators
- `ds_` (damage stats) -- already log1p-normalized

### Global Feature Normalization

| Feature | Method |
|---------|--------|
| `time_norm` | Pre-normalized to [0, 1] |
| Ban IDs | Preserved as categorical integers |
| Differentials (gold, XP, level, CS, alive) | Raw differences (normalized by context) |
| Cumulative objectives | Raw cumulative counts |

---

## 6. Tabular Feature Aggregation

For the LightGBM baseline, temporal sequences are flattened via statistical aggregation.

### Aggregation Suffixes

Defined in `core/feature_contract.py::TABULAR_SUFFIXES`:

```
[last, mean, std, min, max, delta, slope]
```

For each base feature `x` over the L=6 timesteps:

| Suffix | Computation | Description |
|--------|-------------|-------------|
| `__last` | `x[L-1]` | Value at most recent timestep |
| `__mean` | `mean(x)` | Time-averaged value |
| `__std` | `std(x)` | Temporal variability |
| `__min` | `min(x)` | Minimum over window |
| `__max` | `max(x)` | Maximum over window |
| `__delta` | `x[L-1] - x[0]` | Total change over window |
| `__slope` | Least-squares slope | Linear trend |

**Per feature: 7 aggregation values.**

### Enhanced Tabular (with Momentum)

When `USE_MOMENTUM_FEATURES = True`, three additional momentum features are appended:

| Suffix | Computation | Description |
|--------|-------------|-------------|
| `__mu_short` | `(1/k) * sum(dx[-k:])` | Short-term momentum (k=3) |
| `__mu_long` | `mean(dx)` | Long-term momentum |
| `__delta_momentum` | `mu_short - mu_long` | Momentum divergence |

**Per feature: 10 aggregation values (7 base + 3 momentum).**

---

## 7. Feature Constancy Classification

Defined in `core/feature_contract.py`. Used to remove redundant tabular features.

### Strictly Constant Features

Fixed at champion select, never change within a match:

| Category | Features |
|----------|----------|
| **Node (per-player)** | `champion_id`, `champion_name_id`, `summoner_spell_1_id`, `summoner_spell_2_id` |
| **Node (runes)** | `primary_style_id`, `sub_style_id`, `primary_rune_1..4`, `sub_rune_1..2`, `stat_perk_offense/flex/defense` |
| **Global (bans)** | `blue_ban_0..4`, `red_ban_0..4` |

For these features, only `__last` is meaningful. All other suffixes (`__mean`, `__std`, `__min`, `__max`, `__delta`, `__slope`) are either identical to `__last` (redundant) or approximately zero (floating-point noise).

**Impact of removal:** ~16.2% feature reduction.

### Quasi-Constant Features

Extremely unlikely to change within the short teamfight observation window:

| Category | Features | Reason |
|----------|----------|--------|
| **Item hash** | `itemhash*` | Requires recall (impossible mid-fight) |
| **Fight zone** | `zone_top_lane..zone_jungle` | Anchor zone is fixed per engagement |
| **Fight position** | `pos_fight_x_norm`, `pos_fight_y_norm` | Centroid is fixed per engagement |

### Within-Fight Constant Features

Sparse binary signals that are constant within any single observation window but carry meaningful cross-fight signal (handled as a separate category, `WITHIN_FIGHT_CONSTANT_NODE_FEATURE_PREFIXES`):

| Category | Features | Reason |
|----------|----------|--------|
| **Dragon Soul** | `soul_infernal..chemtech` | Once acquired, never changes mid-fight; sparse binary across games |

**Additional reduction:** ~26.2% total when combining all three categories with strictly constant removal.

### Redundancy Removal API

```python
from core.feature_contract import filter_constant_and_quasi_constant

keep_indices, dropped_const, dropped_quasi = filter_constant_and_quasi_constant(
    feature_names,
    drop_strictly_constant=True,
    drop_quasi_constant=True,
    drop_within_fight_constant=True,
)
```

---

## 8. Graph Adjacency Construction

At each time step `t`, an interaction graph is constructed over N=10 player nodes.

### Soft Gaussian Kernel

```
A_ij(t) = exp( -||pos_i(t) - pos_j(t)||^2 / (2 * sigma^2) )
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ADJ_SIGMA_NORM` | 0.125 | Sigma in normalized coordinates (fixed mode) |
| `ADJ_SIGMA_FACTOR` | 1.5 | Scale factor for 60s noise |
| `ADJ_SIGMA_ADAPTIVE` | True | Data-driven sigma per timestep |

**Adaptive sigma:**

```
sigma(t) = 0.5 * d_pair_mean(t)

where d_pair_mean(t) = (1/N^2) * sum_{i,j} ||pos_i(t) - pos_j(t)||
```

### Edge Modifications

| Modification | Config | Description |
|-------------|--------|-------------|
| Team edge upweight | `TEAM_EDGE_WEIGHT = 1.0` | Same-team edges multiplied by factor |
| Dead player masking | `USE_ALIVE_MASK = True` | Dead players' edges set to 0 |
| Self-loops | Always preserved | For numerical stability |

### MPNN Edge Features

For the MPNN model, edge features are explicitly computed:

```
e_ij = [dx, dy, d, log(1 + A_ij)]
```

Where `dx`, `dy` are normalized position differences, `d` is Euclidean distance, and `A_ij` is the Gaussian kernel value.

---

## 9. Spatial Features

Additional spatial features computed per bin for the extra sequence:

| Feature | Description |
|---------|-------------|
| `zone_top_lane` | Binary: fight center in top lane region |
| `zone_mid_lane` | Binary: fight center in mid lane region |
| `zone_bot_lane` | Binary: fight center in bottom lane region |
| `zone_river` | Binary: fight center in river region |
| `zone_jungle` | Binary: fight center in jungle region |
| `pos_fight_x_norm` | Normalized fight centroid X |
| `pos_fight_y_norm` | Normalized fight centroid Y |

---

## 10. Constants and Thresholds

### Map Geometry

| Constant | Value | Description |
|----------|-------|-------------|
| `MAP_MAX` | 16,000 | Summoner's Rift coordinate range (raw map units) |
| `VISION_RADIUS` | 1,200 | Ward vision radius (map units) |
| `VISION_RECENT_SEC` | 90 | Recent vision tracking window |
| `VISION_CNT_DENOM` | 10.0 | Vision count normalization denominator |

### Buff Durations

| Buff | Duration (seconds) |
|------|--------------------|
| Baron Nashor | 180 |
| Elder Dragon | 150 |
| Red Buff | 120 |
| Blue Buff | 120 |

### Cooldowns

| Ability | Cooldown (seconds) |
|---------|-------------------|
| Flash | 300 |

### Dragon Soul Types

6 variants: `infernal`, `ocean`, `mountain`, `cloud`, `hextech`, `chemtech`

### Role Ordering

`["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]`

### Slot Names

```
Blue: bTOP, bJNG, bMID, bBOT, bSUP
Red:  rTOP, rJNG, rMID, rBOT, rSUP
```

---

## 11. Data Statistics (Typical Values)

| Metric | Typical Value |
|--------|---------------|
| Matches (Korean ranked) | Variable (env-configurable) |
| Teamfights per match | 4-6 average |
| Total samples | 200k-300k typical |
| Label balance | ~50/50 (blue/red win) |
| Total tabular features (after aggregation) | 1,200+ |
| Strictly constant features dropped | ~16.2% |
| Total features dropped (const + quasi-const) | ~26.2% |
