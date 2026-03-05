# Experimental Protocol

Complete specification of the ablation study framework, 7 domain-knowledge treatments, statistical testing methodology, evaluation metrics, and known data quality issues.

---

## 1. Ablation Study Protocol

A rigorous 5-phase experimental protocol isolates the contribution of each domain-knowledge improvement.

| Phase | Description | Inputs | Outputs |
|-------|-------------|--------|---------|
| **Phase 1: Baseline** | Reproduce baseline across 5 seeds | All models, default config | AUC_baseline +/- CI |
| **Phase 2: Single-Factor** | For each T_i (i=1..7): measure Delta_i | Baseline + one treatment | Delta_i = AUC(baseline + T_i) - AUC(baseline) |
| **Phase 3: Interactions** | Test pairwise interactions of significant treatments | Significant pairs | Delta_{i,j} - (Delta_i + Delta_j) |
| **Phase 4: Sensitivity** | Sweep hyperparameters of significant treatments | HP grids per treatment | Optimal HP per treatment |
| **Phase 5: Final Test** | Best configuration on held-out test set | Best config from Phase 4 | Final AUC, AP, Brier, ECE |

### Execution

```bash
# Phase 1: Baseline reproduction
python experiment_runner.py --phase 1

# Phase 2: Single-factor treatments
python experiment_runner.py --phase 2 --treatment all

# Phase 2: Specific treatment
python experiment_runner.py --phase 2 --treatment T1

# Paper presets
python runner.py --paper_preset core4_1seed --split_mode patch_holdout
python runner.py --paper_preset core4_optimal --split_mode patch_holdout
```

---

## 2. Seven Domain-Knowledge Treatments

### Treatment T1: Focal Loss

**Motivation:** Teamfights with large gold differentials are trivially predictable and contribute noise gradients. Focal loss focuses learning on hard/close fights where composition and positioning matter most.

**Definition:**

```
L_FL = -alpha_t (1 - p_t)^gamma log(p_t)

where p_t = p  if y=1
      p_t = 1-p if y=0

gamma = 2.0    (focusing parameter)
alpha = 0.25   (class balancing)
```

**Effect:** A sample predicted at `p=0.9` receives weight `(0.1)^2 = 0.01` relative to standard BCE.

**Configuration:**

| Parameter | Default | HP Grid |
|-----------|---------|---------|
| `USE_FOCAL_LOSS` | False | - |
| `FOCAL_GAMMA` | 2.0 | {1, 2, 3} |
| `FOCAL_ALPHA` | 0.25 | - |

**Implementation:** `core/improvements.py::FocalLoss`

---

### Treatment T2: Game Phase Encoding

**Motivation:** LoL matches have three distinct strategic phases with different fight dynamics:
- **Early (0-14 min):** Laning, 1v1/2v2 skirmishes
- **Mid (14-25 min):** Objective contests, rotations, small teamfights
- **Late (25+ min):** Full 5v5 teamfights, death timers are critical

**Definition:**

```
phi(t) = [ sigma((14 - t) / tau),
           sigma((t - 10) / tau) * sigma((28 - t) / tau),
           sigma((t - 22) / tau) ]

tau = 3.0    (transition softness, ~6-minute boundary)
```

Produces a smooth 3-dimensional phase vector `[phi_early, phi_mid, phi_late]` appended to global features.

**Configuration:**

| Parameter | Default | HP Grid |
|-----------|---------|---------|
| `USE_GAME_PHASE` | False | - |
| `GAME_PHASE_TAU` | 3.0 | {2, 3, 4} |

**Implementation:** `core/improvements.py::compute_game_phase_encoding()`

---

### Treatment T3: Temporal Attention Pooling

**Motivation:** Standard last-hidden-state pooling (`h_T`) causes critical early events (power spikes, item completions) to decay through the RNN. Attention pooling allows the model to attend to arbitrary timesteps.

**Definition:**

```
e_t = w^T tanh(W_a h_t + b_a)         (attention energy)
alpha_t = softmax(e_1, ..., e_T)_t     (attention weight)
c = sum_t alpha_t * h_t               (weighted context)

output = [h_T || c]                    (concatenation)
```

**Properties:**
- Attention weights `alpha_t` are interpretable (reveal which timesteps matter)
- Output dimension doubles when `concat_last=True`
- Supports padding masks for variable-length sequences

**Configuration:**

| Parameter | Default | HP Grid |
|-----------|---------|---------|
| `USE_ATTENTION_POOL` | False | - |
| `ATTN_DIM` | 64 | - |

**Implementation:** `core/improvements.py::TemporalAttentionPooling`

---

### Treatment T4: Momentum Features (MACD-Inspired)

**Motivation:** Standard tabular aggregation computes `delta = x[-1] - x[0]` and `slope` but cannot distinguish between long-term trend and recent acceleration. Borrowing from financial technical analysis (MACD), we compute short-term vs. long-term momentum divergence.

**Definition:**

```
dx_t = x_t - x_{t-1}                              (first difference)
mu_short = (1/k) sum_{t=T-k+1}^{T} dx_t           (recent k-step momentum)
mu_long  = (1/(T-1)) sum_{t=2}^{T} dx_t            (overall trend)
delta_momentum = mu_short - mu_long                 (divergence)
```

**Interpretation:**
- `delta_momentum > 0`: Feature is accelerating (recent gains exceed average)
- `delta_momentum < 0`: Feature is decelerating (losing momentum)
- `delta_momentum ~ 0`: Steady trend

**Configuration:**

| Parameter | Default | HP Grid |
|-----------|---------|---------|
| `USE_MOMENTUM_FEATURES` | False | - |
| `MOMENTUM_K_SHORT` | 3 | {3, 5} |

**Implementation:** `core/improvements.py::compute_momentum_features()`

---

### Treatment T5: Role-Aware Adjacency

**Motivation:** Standard distance-based adjacency treats all player pairs equally at the same distance. But strategic relationships between roles matter: jungle-mid synergy is stronger than top-support proximity at equal distances.

**Definition:**

```
A'_ij = A^dist_ij * softplus(R_{role(i), role(j)})

R in R^{5x5}    (learnable role interaction matrix)
init_value = 0.0  (softplus(0) = ln(2) ~ 0.693)
```

**Role order:** `[TOP, JNG, MID, BOT, SUP]`

Since N=10 (5 per team), role indices wrap: `role_idx = player_idx % 5`.

**Configuration:**

| Parameter | Default | HP Grid |
|-----------|---------|---------|
| `USE_ROLE_AWARE_ADJ` | False | - |
| `ROLE_ADJ_INIT` | 0.0 | - |

**Implementation:** `core/improvements.py::RoleAwareAdjacency`

---

### Treatment T6: Multi-Task Auxiliary Losses

**Motivation:** Following Ruder (2017), auxiliary regression targets provide implicit regularization by forcing the shared representation to encode information useful for multiple related tasks.

**Definition:**

```
L = L_fight + lambda_g * ||y_hat_gold - y_gold||^2
            + lambda_k * ||y_hat_kill - y_kill||^2
            + lambda_o * ||y_hat_obj - y_obj||^2

lambda_g = 0.1    (gold differential weight)
lambda_k = 0.05   (kill differential weight)
lambda_o = 0.05   (objective differential weight)
```

**Auxiliary targets:**
- `y_gold`: normalized gold swing in fight window
- `y_kill`: normalized kill differential
- `y_obj`: normalized objective differential

**Architecture:** `MultiTaskHead` adds dedicated regression heads branching from the shared encoder output.

**Configuration:**

| Parameter | Default | HP Grid |
|-----------|---------|---------|
| `USE_MULTI_TASK` | False | - |
| `MT_LAMBDA_GOLD` | 0.1 | {0.05, 0.1, 0.2} |
| `MT_LAMBDA_KILL` | 0.05 | - |
| `MT_LAMBDA_OBJ` | 0.05 | - |

**Implementation:** `core/improvements.py::MultiTaskHead`, `MultiTaskLoss`

---

### Treatment T7: Label Smoothing

**Motivation:** Hard binary labels can cause overconfident predictions. Label smoothing acts as KL-regularization toward the uniform distribution.

**Definition:**

```
y_smooth = y * (1 - epsilon) + epsilon / 2

epsilon = 0.05:
  positive labels: 1.0 -> 0.975
  negative labels: 0.0 -> 0.025
```

**Configuration:**

| Parameter | Default | HP Grid |
|-----------|---------|---------|
| `LABEL_SMOOTHING` | 0.0 | {0.03, 0.05, 0.10} |

**Implementation:** Integrated into `FocalLoss.forward()` and standard BCE.

---

## 3. Statistical Testing

### 3.1 DeLong's Test (AUC Comparison)

For comparing AUC of two models on the same test set (correlated samples):

```
Z = (AUC_A - AUC_B) / sqrt( Var(AUC_A) + Var(AUC_B) - 2*Cov(AUC_A, AUC_B) )
```

Under H0 (equal AUC), Z ~ N(0,1). Two-sided p-value: `p = 2 * (1 - Phi(|Z|))`.

### 3.2 McNemar's Test (Classification Disagreement)

Continuity-corrected chi-squared test on the 2x2 disagreement table:

```
chi^2 = (|b - c| - 1)^2 / (b + c)

where b = samples correct by A but wrong by B
      c = samples correct by B but wrong by A
```

Under H0 (equal accuracy), `chi^2 ~ chi^2(1)`.

### 3.3 Holm-Bonferroni Correction

For family-wise error rate control across `m = 7` treatments at significance level `alpha = 0.05`:

1. Sort p-values: `p_(1) <= p_(2) <= ... <= p_(m)`
2. Reject H_{(i)} if `p_(i) <= alpha / (m - i + 1)`
3. Stop at first non-rejection

This is uniformly more powerful than Bonferroni while maintaining FWER control.

### 3.4 Bootstrap Confidence Intervals

- **Seeds:** {7, 42, 123, 256, 512} (5 independent runs)
- **Resamples:** 1,000 per seed
- **Method:** Percentile (2.5th and 97.5th percentile)
- **Reported:** Mean +/- 95% CI across 5 seeds

---

## 4. Evaluation Metrics

### Primary Metrics

| Metric | Definition | Use |
|--------|-----------|-----|
| **AUC** | Area under the ROC curve | Primary ranking metric |
| **AP** | Area under the Precision-Recall curve | Handles class imbalance |
| **Accuracy** | Correct predictions / total at threshold 0.5 | Intuitive benchmark |

### Calibration Metrics

| Metric | Definition | Use |
|--------|-----------|-----|
| **Brier Score** | `(1/N) sum_i (p_i - y_i)^2` | Proper scoring rule (lower = better) |
| **ECE** | Expected Calibration Error | Binned calibration gap |

### Per-Class Metrics

| Metric | Definition |
|--------|-----------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 * Precision * Recall / (Precision + Recall) |

Reported per-class (blue win / red win) and macro-averaged.

### Subgroup Analysis

| Subgroup | Definition | Rationale |
|----------|-----------|-----------|
| **Early game** | Fights at < 15 minutes | Laning phase, limited info |
| **Mid game** | Fights at 15-25 minutes | Objective contests |
| **Late game** | Fights at > 25 minutes | Full teamfights |
| **Close fight** | abs(gold_diff) < 2,000 | Hard to predict |
| **Moderate** | 2,000 <= abs(gold_diff) <= 5,000 | Typical variance |
| **Stomp** | abs(gold_diff) > 5,000 | Easy to predict |
| **By fight type** | teamfight / skirmish / objective | Structural differences |

---

## 5. Data Splitting Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SPLIT_MODE` | `"multi_patch"` | Split strategy |
| `VAL_FRAC` | 0.20 | Validation set fraction |
| `TEST_FRAC` | 0.10 | Test set fraction |
| `MAX_MATCHES` | None (unlimited) | Optional match count cap |
| `SPLIT_GROUP_BY_MATCH_ID` | True | Prevent match-level leakage |

---

## 6. Recency Weighting

Mitigates patch covariate shift for the LightGBM baseline:

```
w_i = exp( (p_i - p_min) / tau )

tau = 2.0  (default)
```

Where `p_i` is the integer-encoded patch number. Higher `tau` = weaker recency effect.

| Parameter | Default |
|-----------|---------|
| `RECENCY_WEIGHT_ENABLED` | True (patch-drift weighting) |
| `RECENCY_WEIGHT_TAU` | 2.0 |
| `GLOBAL_SUBSAMPLE_PER_SPLIT` | 100,000 |

---

## 7. Post-Hoc Calibration

Per-patch temperature scaling applied after training:

```
P_calibrated = sigmoid(z / T_p*)

T_p* = argmin_T sum_i [ -y_i log sigmoid(z_i/T) - (1-y_i) log(1 - sigmoid(z_i/T)) ]
```

| Parameter | Default |
|-----------|---------|
| `TEMP_SCALING_ENABLED` | False |

---

## 8. Visualization Outputs

The framework automatically generates:

| Plot | Description |
|------|-------------|
| **Forest Plot** | Effect sizes with 95% CI for each treatment |
| **ROC Curves** | Per-model comparison with confidence bands |
| **Reliability Diagram** | Calibration assessment (predicted vs. actual probability) |
| **Interaction Heatmap** | Synergistic/antagonistic treatment interactions |
| **Cumulative Addition Curve** | Sequential model stacking performance |
| **Minute-wise Performance** | AUC by game time (early/mid/late) |
| **Situation-aware Metrics** | Performance by gold state (close/stomp) |

---

## 9. Known Data Quality Issues

### 9.1 Extended Fight Duration (> 60s)

- **Count:** 838 fights (max 111,497 ms ~ 112s)
- **Severity:** P3 (Low)
- **Verdict:** Intended behavior
- **Cause:** Kill clusters form when consecutive kills are within 18s, but total cluster span can exceed 50s (e.g., Baron dance, base siege). Duration = cluster_span + 10s pre-kill offset.
- **Cap:** Fights exceeding `MAX_MERGED_FIGHT_DURATION_MS = 60,000 ms` are rejected.
- **Action:** None required. These represent legitimate extended teamfights.

### 9.2 Engagement Overlap (3.67%)

- **Count:** 11,037 cases
- **Severity:** P1 (High for sequential models only)
- **Verdict:** Intended for i.i.d. models; problematic for sequential models
- **Cause:** Post-merge allows temporal overlap when fights are spatially separated (> 4000 units). Simultaneous fights in different map locations (e.g., toplane skirmish + botlane dive) are genuinely independent.
- **Impact by model type:**
  - i.i.d. (LightGBM): No impact
  - Sequential (RNN, Transformer): Potential label leakage between timesteps
- **Mitigations:**
  - Clipping: `fight[i].label_end_ts = min(fight[i].label_end_ts, fight[i+1].engage_ts)`
  - Masking: Skip overlapping pairs during sequence loss computation
  - Merging: Combine overlapping fights into a single extended event

### 9.3 Early Game Fights (t_start < START_OFFSET_MIN)

- **Count:** 3,598 cases (1.20%)
- **Severity:** P2 (Medium)
- **Verdict:** Bug (intent-vs-implementation mismatch)
- **Cause:** `START_OFFSET_MIN = 2` (minutes) is defined in config but never enforced in the detection pipeline. The actual filter only requires `engage_ts >= t_min + 60,000 ms` (1 minute).
- **Impact:** At 1 minute into the game, feature information is sparse and predictions are unreliable.
- **Recommended fix:** Enforce `START_OFFSET_MIN` in `gameplay/fights.py` or `data/index_split.py`.

### Summary Table

| Issue | Count | Severity | Verdict | Action |
|-------|-------|----------|---------|--------|
| Duration > 60s | 838 | P3 Low | Intended | Document in paper |
| Overlap | 11,037 | P1 High* | Intended (i.i.d.) | Clip/mask for sequential models |
| Early fights | 3,598 | P2 Medium | Bug | Enforce START_OFFSET_MIN |

*P1 only for sequential model architectures.

---

## 10. Reproducibility Checklist

| Item | Status | Details |
|------|--------|---------|
| Fixed random seeds | Yes | {7, 42, 123, 256, 512} via `core/utils.py::seed_everything()` |
| Deterministic operations | Partial | `torch.use_deterministic_algorithms()` not enforced (some ops non-deterministic on GPU) |
| Match-grouped splits | Yes | All fights from one match in same partition |
| Cache versioning | Yes | `CACHE_VERSION` invalidates stale caches |
| Singleton reset | Yes | `reset_model_singletons()` between experiments |
| Environment variables | Yes | Data paths via env vars, no hardcoded paths |
| Configuration snapshot | Yes | `CFG` dataclass serialized with experiment results |
