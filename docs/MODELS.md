# Model Architectures

Complete specification of all 25+ model architectures, their mathematical definitions, hyperparameters, and the Layered Fusion framework.

---

## Architecture Taxonomy

```
+-- Tabular Baseline
|   +-- LightGBM
|   +-- MLP (matched-input tabular; see Section 1.1)
|
+-- Sequential (RNN Family)
|   +-- BiGRU
|   +-- BiLSTM
|   +-- Transformer
|   +-- TCN (Temporal Convolutional Network)
|   +-- Mamba (State-Space Model)
|
+-- Hybrid h0 Conditioning
|   +-- Hybrid BiGRU
|   +-- Hybrid BiLSTM
|
+-- Graph Neural Networks
|   +-- GCN (Graph Convolutional Network)
|   +-- GraphSAGE
|   +-- GATv2 (Graph Attention Network v2)
|   +-- MPNN (Message Passing Neural Network)
|
+-- Spatio-Temporal
|   +-- ST-GNN
|   +-- ST-GCN
|   +-- ST-Mamba
|   +-- Multi-Scale Dynamic Graph (ms_dyngraph)
|   +-- EventXAttn (Event Cross-Attention)
|
+-- Fusion
    +-- Layered Fusion (gated global + graph + event)
    +-- Gated GNN-BiGRU Fusion
```

---

## 1. Tabular Baseline: LightGBM

### Input Representation

The temporal sequences X^node, X^global, X^event are flattened into tabular features via statistical aggregation:

```
phi_tab(x) = [last, mean, std, min, max, delta, slope]
```

Where `delta = x[L] - x[1]` and `slope` is the least-squares linear trend. This produces a high-dimensional tabular vector (see `docs/FEATURES.md` Section 6).

### Recency Weighting

Addresses patch covariate shift:

```
w_i = exp( (p_i - p_min) / tau )
```

Where `p_i` is the integer-encoded patch number and `tau = 2.0` controls recency preference strength.

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 5,000 | Number of boosting rounds |
| `learning_rate` | 0.03 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `num_leaves` | 31 | Maximum number of leaves per tree |
| `subsample` | 0.7 | Row subsampling ratio |
| `colsample_bytree` | 0.7 | Feature subsampling ratio |
| `reg_alpha` | 1.0 | L1 regularization |
| `reg_lambda` | 5.0 | L2 regularization |
| `min_data_in_leaf` | 200 | Minimum data in a leaf |
| `early_stopping_rounds` | 200 | Patience for early stopping |

### Constant Feature Filtering

Before training, the `filter_constant_and_quasi_constant()` function removes redundant tabular features (see `docs/FEATURES.md` Section 7), reducing dimensionality by ~26%.

### 1.1 MLP Baselines

Two MLP baselines exist and must not be confused:

- **Matched-input tabular MLP** (`analysis/mlp_ablation.py::run_mlp_baseline`) — the paper's MLP. It consumes the **same ~2,980-D tabular vector** as LightGBM (built via `build_tabular_Xy`), isolating the learner-family effect from the input representation (paper: MLP .626 vs LightGBM .675).
- **Registry `mlp`** (`train/models.py::MLPTabularModel`, keys `mlp` / `mlp_tab` / `mlp_tabular`) — a diagnostic model inside the deep-training harness. It aggregates the temporal sequence into `[last ‖ mean]` (optionally after an input projection), then applies an MLP head (hidden=256, layers=3 — in-code defaults, overridable via `cfg.MLP_HIDDEN` / `cfg.MLP_LAYERS`). Its input is the 95-D macro sequence, so its scores are **not** comparable to the paper's matched-input MLP.

---

## 2. Sequential Models (RNN Family)

All sequential models process the macro feature sequence S in R^{L x D} (concatenation of flattened node, global, event, and spatial features).

### 2.1 BiGRU / BiLSTM

2-layer bidirectional recurrent networks.

**Architecture:**

```
Input S: (B, L=6, D_input)
    |
    v
InputProjection(D_input, 256) [if USE_INPUT_PROJECTION]
    |
    v
LayerNorm(D_proj)
    |
    v
Bidirectional RNN (GRU or LSTM)
  layers=2, hidden=128, dropout=0.20
    |
    v
h_seq: (B, L, 2*hidden)  -->  h_T = h_seq[:, -1, :]
    |
    v
Classification Head
  Linear(2*hidden, hidden) -> ReLU -> Dropout(0.20) -> Linear(hidden, 1)
    |
    v
logit -> sigmoid -> P(blue wins)
```

**GRU cell equations:**

```
z_t = sigma(W_z x_t + U_z h_{t-1} + b_z)         (update gate)
r_t = sigma(W_r x_t + U_r h_{t-1} + b_r)         (reset gate)
h_hat_t = tanh(W_h x_t + U_h (r_t . h_{t-1}) + b_h)  (candidate)
h_t = (1 - z_t) . h_{t-1} + z_t . h_hat_t        (output)
```

**LSTM cell equations:**

```
f_t = sigma(W_f x_t + U_f h_{t-1} + b_f)         (forget gate)
i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)         (input gate)
o_t = sigma(W_o x_t + U_o h_{t-1} + b_o)         (output gate)
c_hat_t = tanh(W_c x_t + U_c h_{t-1} + b_c)      (candidate cell)
c_t = f_t . c_{t-1} + i_t . c_hat_t               (cell state)
h_t = o_t . tanh(c_t)                              (hidden state)
```

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `bidirectional` | True |
| `dropout` | 0.20 |

### 2.2 Transformer

2-layer self-attention with sinusoidal positional encoding.

**Architecture:**

```
Input S: (B, L=6, D_input)
    |
    v
Linear(D_input, d_model=64)
    |
    v
+ Sinusoidal Positional Encoding (L, d_model)
    |
    v
TransformerEncoder(
  num_layers=2,
  d_model=64,
  nhead=4,
  dim_feedforward=128,   # d_model x TRANS_FF_MULT (2)
  dropout=0.1
)
    |
    v
h_cls = output[:, -1, :]  (last token as summary)
    |
    v
Classification Head
```

**Self-attention:**

```
Q = X W_Q,  K = X W_K,  V = X W_V

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

**Multi-head attention:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
```

| Parameter | Value |
|-----------|-------|
| `d_model` | 64 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dim_feedforward` | 128 |
| `dropout` | 0.1 |

### 2.3 TCN (Temporal Convolutional Network)

3-level causal dilated convolution.

**Architecture:**

```
Input S: (B, L=6, D_input)
    |
    v
Linear(D_input, channels=64) + transpose to (B, C, L)
    |
    v
TemporalBlock(channels=64, kernel=3, dilation=1, dropout=0.20)
  -> causal padding -> Conv1d -> weight_norm -> ReLU -> dropout
  -> Conv1d -> weight_norm -> ReLU -> dropout + residual
    |
TemporalBlock(channels=64, kernel=3, dilation=2, dropout=0.20)
    |
TemporalBlock(channels=64, kernel=3, dilation=4, dropout=0.20)
    |
    v
output[:, :, -1]  (last temporal position)
    |
    v
Classification Head
```

**Causal convolution:**

```
y_t = sum_{k=0}^{K-1} w_k * x_{t - d*k}

where d = dilation factor, K = kernel_size
```

Dilations `[1, 2, 4]` give receptive field = `2 * (K-1) * sum(dilations) + 1 = 2 * 2 * 7 + 1 = 29 > L=6`.

| Parameter | Value |
|-----------|-------|
| `channels` | 64 |
| `levels` | 3 |
| `kernel_size` | 3 |
| `dilations` | [1, 2, 4] |
| `dropout` | 0.20 |

### 2.4 Mamba (State-Space Model)

3-layer selective state-space model.

**State-space equations:**

```
h_t = A_bar h_{t-1} + B_bar x_t       (state transition)
y_t = C h_t + D x_t                    (output)

where A_bar, B_bar are input-dependent (selective mechanism):
  A_bar = exp(delta_t * A)
  B_bar = delta_t * B(x_t)
  delta_t = softplus(Linear(x_t))      (input-dependent step size)
```

| Parameter | Value |
|-----------|-------|
| `d_model` | 128 |
| `num_layers` | 3 |
| `d_state` | 16 |
| `d_conv` | 4 |
| `expand_factor` | 2 |

---

## 3. Hybrid h0 Conditioning

Projects tabular summary features into the initial hidden state of the RNN:

```
h_0 = MLP_tab(phi_tab(S))

where phi_tab(S) = [last, mean, std, min, max, delta, slope] per feature
      MLP_tab: R^{D_tab} -> R^{d_hidden}
```

This provides the RNN with a global context (average gold lead, overall team strength) before processing the temporal sequence.

| Parameter | Default |
|-----------|---------|
| `HYBRID_H0_ENABLED` | True (for hybrid models) |
| `HYBRID_H0_PROJ_DIM` | 64 |
| `HYBRID_H0_DROPOUT` | 0.15 |

---

## 4. Graph Neural Networks

At each time step `t`, an interaction graph G_t = (V, E_t) is constructed over N=10 player nodes.

### Adjacency Matrix

```
A_ij(t) = exp( -||pos_i(t) - pos_j(t)||^2 / (2 sigma^2) )
```

With optional modifications:
- Team edge upweight: `A_ij *= w_team` if same team (`TEAM_EDGE_WEIGHT = 1.0`)
- Dead player masking: `A_ij = 0` if either player is dead (`USE_ALIVE_MASK = True`)
- Self-loops: Always preserved

### 4.1 GCN (ResGCNLayer)

Symmetric normalized graph convolution with residual connections.

```
H' = sigma( D^{-1/2} A_hat D^{-1/2} H W + b )

where A_hat = A + I  (self-loop)
      D_ii = sum_j A_hat_ij
```

With residual: `H_out = LayerNorm(H' + H_in)`

| Parameter | Value |
|-----------|-------|
| `dim` | 96 |
| `dropout` | 0.25 |
| `norm` | LayerNorm |
| `activation` | ReLU |

### 4.2 GraphSAGE

Mean neighbor aggregation.

```
h_i' = W * [h_i || mean_{j in N(i)} h_j] + b
```

Degree-normalized: neighbors are averaged, not summed, making the aggregation invariant to node degree.

| Parameter | Value |
|-----------|-------|
| `dim` | 96 |
| `dropout` | 0.25 |
| `norm` | LayerNorm |

### 4.3 GATv2 (Graph Attention Network v2)

Multi-head attention with hard adjacency masking.

```
alpha_ij = softmax_j( a^T LeakyReLU(W [h_i || h_j]) )

h_i' = sigma( sum_j alpha_ij * W_v h_j )
```

**Hard adjacency masking:** When `A_ij = 0` (no edge), attention score is forced to 0. This prevents spurious attention to disconnected nodes.

Multi-head: 4 heads, concatenated (or averaged for final layer).

| Parameter | Value |
|-----------|-------|
| `dim` | 96 |
| `heads` | 4 |
| `dropout` | 0.25 |
| `leaky_alpha` | 0.2 |
| `hard_mask` | True (A=0 -> attention=0) |

### 4.4 MPNN (Message Passing Neural Network)

Edge-conditioned message passing.

```
m_ij = MLP_e(e_ij) . (W h_j)       (message)
h_i' = sigma( sum_j m_ij )          (aggregation)

where e_ij = [dx, dy, d, log(1+A)]  (edge features, dim=4)
```

Edge features: normalized position differences, Euclidean distance, and log-transformed adjacency weight.

| Parameter | Value |
|-----------|-------|
| `edge_dim` | 4 |
| `hidden` | 128 |

---

## 5. Spatio-Temporal Models

### 5.1 ST-GNN

GNN applied independently at each timestep, followed by temporal pooling.

```
For each t in {1, ..., L}:
    h_t = GNN(node_features_t, A_t)     # (N, d_gnn)

h_graph = TemporalPool(h_1, ..., h_L)   # attention or mean
    |
    v
Classification Head
```

### 5.2 ST-GCN

Interleaved graph convolution and temporal convolution blocks.

```
X: (B, L, N, D)

For each block:
    X = GraphConv(X, A)    # spatial mixing (per timestep)
    X = TemporalConv(X)    # temporal mixing (per node)
    X = ReLU + Dropout
```

### 5.3 ST-Mamba

GNN spatial processing combined with Mamba state-space temporal modeling.

```
For each t:
    h_t = GNN(node_features_t, A_t)     # spatial

h_seq = stack(h_1, ..., h_L)            # (B, L, d)
output = Mamba(h_seq)                     # temporal (selective SSM)
```

### 5.4 Multi-Scale Dynamic Graph (ms_dyngraph)

Multiple Gaussian kernels at different sigma values, weighted and summed.

```
A = sum_k w_k * A(sigma_k)

where w_k are learnable weights
      sigma_k covers multiple spatial scales
```

This captures both close-range interactions (small sigma) and long-range awareness (large sigma) simultaneously.

### 5.5 EventXAttn (Event Cross-Attention)

Processes discrete event tokens through cross-attention with graph-encoded player states.

**Three stages:**

**Stage 1: Event Embedding**

```
e_i = Embed(type_hash_i) + Linear(continuous_features_i)
```

**Stage 2: Cross-Attention**

```
XAttn(Q_event, K_graph, V_graph) = softmax( Q K^T / sqrt(d) ) V

where Q = event embeddings, K/V = graph-encoded player states
```

**Stage 3: Importance-Weighted Pooling**

```
h_event = sum_i w_i * e_i

where w_i incorporates learned importance + domain prior
      (shutdowns, objectives weighted higher)
```

---

## 6. Layered Fusion Architecture

The core contribution: a gated fusion model unifying three encoding streams.

### Architecture Diagram

```
              +-- Global Stream -------------------------+
              |  extra_seq -> BiGRU/Transformer -> h_global  |
              +-- Graph Stream --------------------------+
              |  node_seq + A(t) -> GraphSAGE/GAT -> h_graph |
              +-- Event Stream --------------------------+
              |  event_tokens -> XAttn(events, graph) -> h_event |
              +---------------------------------------------+
                            |
                            v
              +-- Gated Fusion Layer --------------------+
              |  h_cat = [h_global || h_graph || h_event]    |
              |  g = sigma(W_g * h_cat + b_g)                |
              |  h_fuse = g . MLP(h_cat)                     |
              +---------------------------------------------+
                            |
                            v
              +-- Optional: Baseline Logit Stacking -----+
              |  h_final = [h_fuse || logit_lgbm]           |
              +---------------------------------------------+
                            |
                            v
                  MLP Head -> logit -> sigmoid -> P(blue wins)
```

### Gated Fusion Equations

```
g = sigma( W_g [h_global || h_graph || h_event] + b_g )
h_fuse = g . MLP( [h_global || h_graph || h_event] )
```

The gate vector `g` learns to weight contributions from different modalities. The element-wise product allows selective amplification or suppression of each feature dimension.

### Configurable Components

| Stream | Options |
|--------|---------|
| **Global encoder** | UniGRU, BiGRU, UniLSTM, BiLSTM, Transformer, TCN, Mamba |
| **GNN encoder** | GCN, GraphSAGE, GraphTransformer, GATv2, MPNN |
| **Event encoder** | Self-attention, Cross-attention, Mean pooling |
| **Baseline logit** | With / without LightGBM logit passthrough |

This yields a configurable combinatorial search space for architecture selection.

### Fusion Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `fuse_dim` | 192 | Fusion hidden dimension |
| `gate_h` | 64 | Gate projection dimension |
| `event_d_model` | 128 | Event encoder model dimension |
| `FUSION_GATE_H` | 8 | Gated fusion hidden dimension |
| `FUSION_MLP_H` | 32 | Fusion MLP hidden dimension |

---

## 7. Node Feature Adapter

Handles the heterogeneous node feature vector (mix of continuous scalars and categorical IDs).

```
Input: node_seq (B, L, N, F_node)
    |
    +-- champion_id     -> Embedding(num_champs, d_embed)  --+
    +-- champion_name_id -> Embedding(num_names, d_embed)   --+
    +-- rune_ids        -> Embedding(num_runes, d_embed)    --+-- concat
    +-- spell_ids       -> Embedding(num_spells, d_embed)   --+
    +-- numeric_features -> Linear(F_numeric, d_numeric)    --+
                                                               |
                                                               v
                                                   (B, L, N, d_hidden)
```

The `NodeFeatureAdapter` (in `train/node_adapter.py`) separates categorical columns from numeric columns, applies learned embeddings to categoricals, projects numerics through a linear layer, and concatenates the results.

---

## 8. Temporal Pooling Strategies

### Last Hidden State (default)

```
h_out = h_seq[:, -1, :]     # (B, d_hidden)
```

### Temporal Attention Pooling (Treatment T3)

```
e_t = w^T tanh(W_a h_t + b_a)          (attention energy)
alpha_t = softmax(e_1, ..., e_T)_t      (attention weight)
c = sum_t alpha_t * h_t                 (context vector)

output = [h_T || c]                      (concat_last mode: dim doubles)
```

Allows the model to attend to critical moments in the pre-fight sequence.

### Mean Pooling

```
h_out = (1/L) sum_t h_t
```

### Team-Aware Graph Pooling

```
h_blue = pool(h_0, h_1, h_2, h_3, h_4)     # blue team nodes
h_red  = pool(h_5, h_6, h_7, h_8, h_9)     # red team nodes
h_out  = [h_blue || h_red]                   # (2 * d_gnn)
```

Implemented in `train/graph_encoder.py::pool_team_repr()`.

---

## 9. Loss Functions

### Binary Cross-Entropy (default)

```
L_BCE = -[y log(p) + (1-y) log(1-p)]

where p = sigmoid(logit)
```

### Focal Loss (Treatment T1)

```
L_FL = -alpha_t (1 - p_t)^gamma log(p_t)

gamma = 2.0    (focusing parameter)
alpha = 0.25   (class balancing)
```

A sample predicted at `p = 0.9` correctly receives weight `(0.1)^2 = 0.01` relative to BCE.

### Multi-Task Loss (Treatment T6)

```
L_total = L_fight + lambda_g * ||y_hat_gold - y_gold||^2
                  + lambda_k * ||y_hat_kill - y_kill||^2
                  + lambda_o * ||y_hat_obj - y_obj||^2

lambda_g = 0.1,  lambda_k = 0.05,  lambda_o = 0.05
```

Optional uncertainty weighting (Kendall et al., 2018):

```
L_total = (1/2*sigma_1^2) * L_1 + (1/2*sigma_2^2) * L_2 + (1/2*sigma_3^2) * L_3
        + log(sigma_1 * sigma_2 * sigma_3)
```

### Label Smoothing (Treatment T7)

```
y_smooth = y * (1 - epsilon) + epsilon / 2

epsilon = 0.05  ->  positive: 0.975, negative: 0.025
```

---

## 10. Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | AdamW | With weight decay |
| **Learning rate** | 5e-4 | |
| **Weight decay** | 2e-4 | L2 regularization |
| **Batch size** | 64 | GPU-optimized |
| **Epochs** | 15 | Maximum training epochs |
| **Patience** | 3 | Early stopping epochs |
| **Warmup epochs** | 1 | Linear warmup duration |
| **Gradient clipping** | 5.0 | Max gradient norm |
| **Mixed precision** | AMP (bf16/fp16) | Auto-selected per GPU |
| **Seeds** | {7, 42, 123} | 3-seed replication (`CFG.SEEDS`) |
| **Early stop metric** | Validation AUC | |
| **DEEP_MAX_TRAIN** | 100,000 | Maximum training samples for deep models |
| **GLOBAL_SUBSAMPLE_PER_SPLIT** | 100,000 | Uniform cap on all splits |

### GPU Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TF32` | True | TensorFloat-32 on Ampere GPUs |
| `CUDNN_BENCHMARK` | True | cuDNN auto-tuning |
| `TORCH_COMPILE` | False | Kernel fusion (optional) |
| `SPEED_PROFILE` | "none" | Options: rtx50, rtx5080, aggressive |
| `NUM_WORKERS` | 4 | DataLoader workers |

---

## 11. Model Registry Summary

| Model Key | Category | Key Hyperparameters |
|-----------|----------|---------------------|
| `lgbm` | Baseline | n_estimators=5000, lr=0.03, max_depth=6, num_leaves=31 |
| `mlp` | Baseline (diagnostic) | [last ‖ mean] aggregation, hidden=256, layers=3 (see Section 1.1) |
| `rnn_ugru` | Sequential | hidden=128, layers=2, dropout=0.20, unidirectional |
| `rnn_bigru` | Sequential | hidden=128, layers=2, dropout=0.20 |
| `rnn_ulstm` | Sequential | hidden=128, layers=2, dropout=0.20, unidirectional |
| `rnn_bilstm` | Sequential | hidden=128, layers=2, dropout=0.20 |
| `rnn_transformer` | Sequential | d_model=64, nhead=4, layers=2, dropout=0.1 |
| `rnn_tcn` | Sequential | channels=64, levels=3, kernel=3, dropout=0.20 |
| `rnn_mamba` | Sequential | d_model=128, layers=3, d_state=16, d_conv=4 |
| `hybrid_bigru` | Hybrid | h0_proj_dim=64, h0_dropout=0.15 |
| `hybrid_bilstm` | Hybrid | h0_proj_dim=64, h0_dropout=0.15 |
| `hybrid_ugru` | Hybrid | h0_proj_dim=64, h0_dropout=0.15 |
| `gnn_gcn` | Graph | dim=96, dropout=0.25, norm=LayerNorm |
| `gnn_graphsage` | Graph | dim=96, dropout=0.25, norm=LayerNorm |
| `gnn_graphtransformer` | Graph | dim=96, dropout=0.25, norm=LayerNorm |
| `gnn_gatv2` | Graph | dim=96, heads=4, dropout=0.25, leaky_alpha=0.2 |
| `gnn_mpnn` | Graph | edge_dim=4, hidden=128 |
| `gnn_stgnn` | Spatio-Temporal | GNN per timestep + temporal GRU |
| `gnn_stgcn` | Spatio-Temporal | GNN spatial + TCN temporal |
| `edge_stgnn` | Spatio-Temporal | Edge-augmented MPNN spatial + GRU temporal |
| `stgnn_mamba` | Spatio-Temporal | GNN spatial + Mamba temporal |
| `ms_stgnn` | Spatio-Temporal | Multiscale adjacency + EdgeSTGNN |
| `ms_stgcn` | Spatio-Temporal | Multiscale adjacency + STGCN |
| `event_xattn` | Spatio-Temporal | Cross-attention events over graph states |
| `fusion_gated_gnn_bigru` | Fusion | gate_h=8, mlp_h=32 |
| `fusion_layered_gnn_bigru_xattn` | Fusion | fuse_dim=192, gate_h=64, event_d_model=128 |
