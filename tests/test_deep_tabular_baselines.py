"""Tests for the ToG A1 deep tabular baselines and the declared hyper-parameter search.

scripts/run_deep_tabular_baselines.py and scripts/run_deep_hparam_search_v33.py:

  1  TabNet: the sparsity term is ADDED (larger lambda -> lower mask entropy; the published sign
     raised it), sparsemax is the simplex projection, ghost BN normalises virtual batches
  2  SAINT pre-training: the loss decreases on a toy, and pre-training sees only training rows
  3  scaled_dot_product_attention equals explicit softmax attention (fp32 max abs diff < 1e-4),
     on CPU and on CUDA when a GPU is present; the published architectures reproduce the CoG
     modules bit for bit; padded bf16 SDPA never forms the token matrix; checkpointing is exact
  5  per-model prediction files hold y, pred, groups in test-patch row order
  plus the published parameter counts, the lr schedules, AdamW parameter groups, loading with
  draws dropped in place, pairing against earlier predictions only on identical test rows, and the
  search protocol (selection on the validation patch only, one test evaluation per family, resume).

Everything but the CUDA tests runs on CPU with tiny models and synthetic shards.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SUFFIXES = ("mean", "std", "min", "max", "first", "last", "slope")
# CUDA_VISIBLE_DEVICES="" can leave is_available() True with no usable device
CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # the search script imports the baselines module by this name
    spec.loader.exec_module(module)
    return module


dtb = _load("run_deep_tabular_baselines")
search = _load("run_deep_hparam_search_v33")


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(4, previous))
    yield
    torch.set_num_threads(previous)


# --------------------------------------------------------------------------- synthetic shards
def make_shards(root: Path, n_bases: int = 3, matches_per_patch: int = 30,
                rows_per_match: int = 3, seed: int = 0) -> Path:
    """Two shards in the corpus_shards_v33 layout: suffix-major names plus frame_age_s."""
    rng = np.random.default_rng(seed)
    names = [f"base{b}__{s}" for s in SUFFIXES for b in range(n_bases)] + ["frame_age_s"]
    X, y, groups, patch = [], [], [], []
    for p in ("15.14", "15.15", "15.16"):
        for m in range(matches_per_patch):
            for _ in range(rows_per_match):
                x = rng.normal(size=len(names)).astype(np.float32)
                label = int(rng.random() < 1.0 / (1.0 + math.exp(-(1.5 * x[0] - x[n_bases]))))
                X.append(x)
                y.append(-1 if rng.random() < 0.05 else label)
                groups.append(f"KR_{p.replace('.', '')}{m:04d}")
                patch.append(p)
    X, y = np.stack(X), np.array(y, dtype=np.int8)
    groups, patch = np.array(groups), np.array(patch)
    order = rng.permutation(len(y))
    root.mkdir(parents=True, exist_ok=True)
    for i, rows in enumerate(np.array_split(order, 2)):
        np.savez_compressed(root / f"shard_{i:03d}.npz", X=X[rows], y=y[rows],
                            y_market_event=y[rows], groups=groups[rows], patch=patch[rows],
                            cluster_blue=rng.integers(1, 6, len(rows)).astype(np.int16),
                            cluster_red=rng.integers(1, 6, len(rows)).astype(np.int16))
    (root / "feature_names.json").write_text(
        json.dumps({"feature_set": "synthetic", "names": names}), encoding="utf-8")
    return root


def labelled_rows(shards: Path) -> dict:
    """Rows loaded everything-first and filtered afterwards (the published loader), draws dropped;
    both scripts must reproduce exactly these rows in exactly this order."""
    data = dtb.load_subsample(shards, None, dtb.SEED, y_key="y_market_event")
    assert data.pop("n_unlabelled_dropped") == 0  # nothing filtered while loading
    keep = data["y"] >= 0
    return {k: v[keep] for k, v in data.items()}


@pytest.mark.parametrize("n_matches", [None, 40])
def test_dropping_unlabelled_rows_while_loading_keeps_rows_and_order(tmp_path, n_matches):
    shards = make_shards(tmp_path / "shards")
    after = dtb.load_subsample(shards, n_matches, dtb.SEED, y_key="y_market_event")
    keep = after["y"] >= 0
    assert not keep.all()  # the synthetic shards do hold draws
    during = dtb.load_subsample(shards, n_matches, dtb.SEED, y_key="y_market_event",
                                drop_unlabelled=True)
    assert during["n_unlabelled_dropped"] == int((~keep).sum())
    for k in ("X", "y", "groups", "patch", "min_participants"):
        assert np.array_equal(during[k], after[k][keep]), k
    assert during["X"].dtype == np.float32


# --------------------------------------------------------------------------- published shapes
def test_published_architectures_keep_their_parameter_counts():
    # deep_tabular_v33_patch_{full,ft,saint}.json: 7,106 columns, 1,015 bases + frame_age_s
    assert dtb.count_params(dtb.build_mlp(7106, 512, 3, 0.1)) == 4_167_681
    assert dtb.count_params(dtb.build_tabnet(7106, 4, 64, 64, 1.3, 0.1)) == 4_075_989
    assert dtb.count_params(dtb.build_ft_transformer(1016, 7, 32, 3, 8, 0.1)) == 285_857
    assert dtb.count_params(dtb.build_saint(1016, 7, 32, 3, 8, 0.1)) == 459_713


def _cog_ft_transformer(n_tokens, n_stats, d_token, n_layers, n_heads, dropout):
    """The CoG-submission FT-Transformer (git 342e74c, scripts/run_deep_tabular_baselines.py),
    copied verbatim apart from formatting: the reference for 'the old attention'."""
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_token)
            self.qkv = nn.Linear(d_token, 3 * d_token)
            self.proj = nn.Linear(d_token, d_token)
            self.norm2 = nn.LayerNorm(d_token)
            self.ff = nn.Sequential(nn.Linear(d_token, d_token * 2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d_token * 2, d_token))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, t, d = x.shape
            h = self.norm1(x)
            q, k, v = self.qkv(h).reshape(b, t, 3, n_heads, d // n_heads).permute(2, 0, 3, 1, 4)
            attended = nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout if self.training else 0.0)
            x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(b, t, d)))
            return x + self.drop(self.ff(self.norm2(x)))

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_tokens, n_stats, d_token))
            self.bias = nn.Parameter(torch.zeros(n_tokens, d_token))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            nn.init.normal_(self.cls, std=0.02)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layers))
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, 1))

        def forward(self, x):
            tokens = torch.einsum("bts,tsd->btd", x, self.weight) + self.bias
            tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
            for block in self.blocks:
                tokens = block(tokens)
            return self.head(tokens[:, 0]).squeeze(-1)

    return FTTransformer()


def _cog_saint(n_tokens, n_stats, d_token, n_layers, n_heads, dropout, d_misa=128):
    """The CoG-submission SAINT (git 342e74c), copied verbatim apart from formatting."""
    nn = torch.nn

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d_token)
            self.qkv = nn.Linear(d_token, 3 * d_token)
            self.proj = nn.Linear(d_token, d_token)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, t, d = x.shape
            q, k, v = self.qkv(self.norm(x)).reshape(
                b, t, 3, n_heads, d // n_heads).permute(2, 0, 3, 1, 4)
            out = nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout if self.training else 0.0)
            return x + self.drop(self.proj(out.transpose(1, 2).reshape(b, t, d)))

    class IntersampleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d_token)
            self.summary = nn.Linear(d_token, d_misa)
            self.qkv = nn.Linear(d_misa, 3 * d_misa)
            self.proj = nn.Linear(d_misa, d_token)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, t, d = x.shape
            pooled = self.summary(self.norm(x).mean(dim=1))
            q, k, v = self.qkv(pooled).reshape(b, 3, n_heads, d_misa // n_heads).permute(1, 0, 2, 3)
            out = nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2), dropout_p=dropout if self.training else 0.0)
            context = self.proj(out.transpose(1, 2).reshape(b, d_misa))
            return x + self.drop(context).unsqueeze(1)

    class SAINT(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_tokens, n_stats, d_token))
            self.bias = nn.Parameter(torch.zeros(n_tokens, d_token))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            nn.init.normal_(self.cls, std=0.02)
            self.self_attn = nn.ModuleList(SelfAttention() for _ in range(n_layers))
            self.inter_attn = nn.ModuleList(IntersampleAttention() for _ in range(n_layers))
            self.ff = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, d_token * 2),
                              nn.GELU(), nn.Dropout(dropout), nn.Linear(d_token * 2, d_token))
                for _ in range(n_layers))
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, 1))

        def forward(self, x):
            tokens = torch.einsum("bts,tsd->btd", x, self.weight) + self.bias
            tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
            for msa, misa, ff in zip(self.self_attn, self.inter_attn, self.ff):
                tokens = msa(tokens)
                tokens = misa(tokens)
                tokens = tokens + ff(tokens)
            return self.head(tokens[:, 0]).squeeze(-1)

    return SAINT()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["ft", "saint"])
def test_published_arch_reproduces_the_cog_modules(kind, device):
    """--ft-arch/--saint-arch published, fp32: the refactor onto _attend (and the new math
    reference) leaves the CoG model unchanged -- same parameter names, same outputs."""
    if device == "cuda" and not CUDA:
        pytest.skip("no CUDA device")
    n_tokens, d = (1016, 32) if device == "cuda" else (40, 16)
    heads = 8 if device == "cuda" else 4
    torch.manual_seed(0)
    if kind == "ft":
        new, old = dtb.build_ft_transformer(n_tokens, 7, d, 3, heads, 0.1), \
            _cog_ft_transformer(n_tokens, 7, d, 3, heads, 0.1)
    else:
        new, old = dtb.build_saint(n_tokens, 7, d, 3, heads, 0.1), \
            _cog_saint(n_tokens, 7, d, 3, heads, 0.1)
    old.load_state_dict(new.state_dict(), strict=True)
    math_ref = (dtb.build_ft_transformer(n_tokens, 7, d, 3, heads, 0.1, attn_impl="math")
                if kind == "ft" else dtb.build_saint(n_tokens, 7, d, 3, heads, 0.1,
                                                      attn_impl="math"))
    math_ref.load_state_dict(new.state_dict(), strict=True)
    x = torch.randn(6, n_tokens, 7, generator=torch.Generator().manual_seed(1)).to(device)
    outs = []
    for model in (new, old, math_ref):
        model.to(device).eval()
        with torch.no_grad():
            outs.append(model(x).float().cpu())
    assert torch.equal(outs[0], outs[1])  # same kernel call, so bit-identical
    assert (outs[0] - outs[2]).abs().max().item() < 1e-4  # vs explicit softmax attention
    if device == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- 1. TabNet
def test_sparsemax_is_the_euclidean_projection_onto_the_simplex():
    p = dtb._sparsemax(torch.tensor([[1.0, 0.8, 0.1], [0.0, 0.0, 0.0]]))
    assert torch.allclose(p, torch.tensor([[0.6, 0.4, 0.0], [1 / 3, 1 / 3, 1 / 3]]), atol=1e-6)
    z = torch.randn(64, 20, generator=torch.Generator().manual_seed(0)) * 3
    p = dtb._sparsemax(z)
    assert (p >= 0).all() and torch.allclose(p.sum(-1), torch.ones(64), atol=1e-5)
    for zr, pr in zip(z, p):  # KKT: p = max(z - tau, 0) with one threshold per row
        support = pr > 0
        tau = (zr - pr)[support]
        assert torch.allclose(tau, tau[0].expand_as(tau), atol=1e-5)
        assert (zr[~support] <= tau[0] + 1e-5).all()


def _tabnet_mask_entropy(lam: float, legacy: bool) -> float:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1024, 32)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    torch.manual_seed(0)
    model = dtb.build_tabnet(32, n_steps=3, n_d=8, n_a=8, gamma=1.3, dropout=0.0)
    out = dtb.train_torch(model, X[:768], y[:768], X[768:], y[768:], None, None, epochs=6,
                          batch_size=128, lr=2e-2, weight_decay=0.0, patience=6, tokenized=False,
                          label="toy", sparsity_lambda=lam, legacy_sparsity_sign=legacy,
                          device="cpu", seed=0)
    assert out["test_auc"] is None and out["_pred"] is None  # no test rows given, none scored
    return out["history"][-1]["mask_entropy"]


def test_tabnet_sparsity_regulariser_is_added_to_the_loss():
    """Arik & Pfister (AAAI 2021) Sec. 3: L = L_task + lambda_sparse * L_sparse with L_sparse the
    mask entropy, so a larger lambda must give sparser masks.  The published run subtracted it."""
    none, weak, strong = (_tabnet_mask_entropy(lam, legacy=False) for lam in (0.0, 0.1, 1.0))
    legacy = _tabnet_mask_entropy(1.0, legacy=True)
    assert strong < weak < none
    assert legacy > none + 0.3


def test_ghost_batch_norm_normalises_each_virtual_batch_but_not_the_input():
    model = dtb.build_tabnet(12, n_steps=2, n_d=4, n_a=4, gamma=1.3, dropout=0.0,
                             virtual_batch_size=4, bn_momentum=0.3)
    assert type(model.bn) is torch.nn.BatchNorm1d  # paper: plain BN on the input features
    gbn = model.attention[0][1]  # attentive transformer h_i = FC -> (ghost) BN
    assert type(gbn).__name__ == "GhostBatchNorm1d"
    x = torch.randn(10, 12, generator=torch.Generator().manual_seed(0))
    gbn.train()
    out = gbn(x)
    for rows in torch.tensor_split(torch.arange(10), math.ceil(10 / 4)):  # sizes 4, 3, 3
        expected = torch.nn.functional.batch_norm(x[rows], None, None, gbn.weight, gbn.bias,
                                                  training=True, eps=gbn.eps)
        assert torch.allclose(out[rows], expected, atol=1e-5)
    plain = dtb.build_tabnet(12, 2, 4, 4, 1.3, 0.0, virtual_batch_size=0).attention[0][1].train()
    expected = torch.nn.functional.batch_norm(x, None, None, plain.weight, plain.bias,
                                              training=True, eps=plain.eps)
    assert torch.allclose(plain(x), expected, atol=1e-5)


def test_tabnet_paper_arch_resolves_to_the_forest_cover_configuration():
    args = dtb.build_parser().parse_args(["--shards", ".", "--output", "x.json",
                                          "--tabnet-arch", "paper"])
    cfg = dtb.published_configs(args)["tabnet"]
    forest = dtb.TABNET_FOREST_COVER
    for key in ("n_d", "n_a", "n_steps", "gamma", "lambda_sparse", "virtual_batch_size",
                "bn_momentum", "lr", "lr_schedule", "weight_decay", "dropout", "init"):
        assert cfg[key] == forest[key], key
    assert cfg["batch_size"] == 4096  # declared deviation from B = 16,384
    deviations = dtb.model_deviations("tabnet", cfg, dtb.training_options(args))
    assert any("batch_size = 4096 (paper 16384)" in d for d in deviations)
    assert not any("plain BatchNorm" in d or "SUBTRACTED" in d for d in deviations)
    published = dtb.published_configs(dtb.build_parser().parse_args(
        ["--shards", ".", "--output", "x.json"]))["tabnet"]
    assert (published["n_steps"], published["gamma"], published["lr"], published["batch_size"],
            published["virtual_batch_size"], published["dropout"]) == (4, 1.3, 2e-3, 256, 0, 0.1)
    model = dtb.build_tabnet(20, cfg["n_steps"], 8, 8, cfg["gamma"], cfg["dropout"],
                             virtual_batch_size=cfg["virtual_batch_size"], init=cfg["init"])
    assert all(m.bias is None or bool((m.bias == 0).all())
               for m in model.modules() if isinstance(m, torch.nn.Linear))


# --------------------------------------------------------------------------- 2. SAINT pre-training
def _saint_toy(rows: int = 512, n_bases: int = 5, n_extras: int = 1):
    rng = np.random.default_rng(0)
    width = 7 * n_bases + n_extras
    X = rng.normal(size=(rows, 4)) @ rng.normal(size=(4, width)) + 0.3 * rng.normal(
        size=(rows, width))
    return ((X - X.mean(0)) / X.std(0)).astype(np.float32), n_bases, n_extras


def _pretrain_toy(arch: str, **overrides):
    X, n_bases, n_extras = _saint_toy()
    torch.manual_seed(0)
    model = dtb.build_saint(n_bases + n_extras, 7, 16, 1, 4, 0.0, d_misa=8, arch=arch,
                            ff_dropout=0.0)
    heads = dtb.build_saint_pretrain_heads(n_bases + n_extras, 7, 16,
                                           dtb.token_stat_mask(n_bases, n_extras))
    kw = dict(epochs=8, batch_size=64, lr=3e-3, weight_decay=0.0, tau=0.7, lambda_denoise=10.0,
              p_cutmix=0.3, mixup_alpha=0.2, input_transform=dtb.make_token_transform(
                  n_bases, n_extras), device="cpu", label="toy")
    kw.update(overrides)
    return dtb.pretrain_saint(model, heads, X, **kw)["history"]


@pytest.mark.parametrize("arch", ["published", "paper"])
def test_saint_pretraining_loss_decreases_on_a_toy(arch):
    # the paper's augmentations (CutMix 0.3, mixup alpha 0.2) keep InfoNCE close to log(batch) on
    # a toy; each term must still fall (8 CPU epochs: total -2% / -5%, each term by > 0.02)
    history = _pretrain_toy(arch)
    first, last = history[0], history[-1]
    assert last["loss"] < 0.99 * first["loss"]
    assert last["denoise"] < first["denoise"] - 0.01
    assert last["contrastive"] < first["contrastive"] - 0.01


def test_saint_contrastive_term_learns_when_the_views_coincide():
    # p_cutmix = 0 and alpha = 1 make the augmented view the original: InfoNCE must fall well
    # below its chance value log(batch) once g1 and g2 align
    history = _pretrain_toy("paper", lambda_denoise=0.0, p_cutmix=0.0, mixup_alpha=1.0)
    assert history[0]["contrastive"] > math.log(64) - 0.5
    assert history[-1]["contrastive"] < math.log(64) - 1.0


def test_denoise_reductions_differ_by_the_column_count():
    X, n_bases, n_extras = _saint_toy(rows=16)
    torch.manual_seed(0)
    model = dtb.build_saint(n_bases + n_extras, 7, 16, 1, 4, 0.0, d_misa=8, arch="paper").eval()
    heads = dtb.build_saint_pretrain_heads(n_bases + n_extras, 7, 16,
                                           dtb.token_stat_mask(n_bases, n_extras)).eval()
    x = dtb.make_token_transform(n_bases, n_extras)(torch.from_numpy(X))
    terms = {}
    for reduction in ("mean", "sum"):
        with torch.no_grad():
            terms[reduction] = dtb.saint_pretrain_loss(
                model, heads, x, tau=0.7, lambda_denoise=10.0, p_cutmix=0.3, mixup_alpha=0.2,
                reduction=reduction, generator=torch.Generator().manual_seed(0))[2]
    assert torch.allclose(terms["sum"], terms["mean"] * (7 * n_bases + n_extras), rtol=1e-5)


def test_saint_pretraining_sees_only_training_rows(monkeypatch):
    rng = np.random.default_rng(0)
    n_bases, n_extras = 3, 1
    X = rng.normal(size=(300, 7 * n_bases + n_extras)).astype(np.float32)
    y = rng.integers(0, 2, 300)
    groups = np.repeat(np.arange(100), 3)
    tr, va = np.arange(0, 120), np.arange(120, 210)
    seen = {}

    def fake_pretrain(model, heads, X_rows, **kw):
        seen["rows"] = np.asarray(X_rows.idx)
        seen["groups"] = kw["groups"]
        return {"history": [], "seconds": 0.0}

    monkeypatch.setattr(dtb, "pretrain_saint", fake_pretrain)
    args = dtb.build_parser().parse_args([
        "--shards", ".", "--output", "x.json", "--d-token", "8", "--n-heads", "2",
        "--n-layers", "1", "--saint-d-misa", "8", "--saint-pretrain-epochs", "1",
        "--saint-pretrain-rows", "50", "--epochs", "1", "--device", "cpu"])
    cfg, opts = dtb.published_configs(args)["saint"], dtb.training_options(args)
    entry = dtb.fit_model("saint", cfg, opts, X, y, tr, va, None, (n_bases, n_extras),
                          groups=groups)
    assert len(seen["rows"]) == 50 and set(seen["rows"].tolist()) <= set(tr.tolist())
    assert np.array_equal(seen["groups"], groups[seen["rows"]])
    assert entry["test_auc"] is None and entry["pretrain"] is not None
    with pytest.raises(ValueError, match="match id"):  # no silent fallback to mixed batches
        dtb.fit_model("saint", cfg, opts, X, y, tr, va, None, (n_bases, n_extras))


def test_match_disjoint_batches_never_share_a_match():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(120), rng.integers(1, 10, 120))
    batches = dtb.match_disjoint_batches(groups, 32, np.random.default_rng(1))
    rows = np.concatenate(batches)
    assert np.array_equal(np.sort(rows), np.arange(len(groups)))  # every row exactly once
    assert all(len(np.unique(groups[b])) == len(b) for b in batches)
    sizes = [len(b) for b in batches]
    assert max(sizes) - min(sizes) <= 1 and len(batches) == math.ceil(len(groups) / 32)
    tiny = dtb.match_disjoint_batches(np.array([5, 5, 5, 7, 7]), 1024, np.random.default_rng(0))
    assert len(tiny) == 3 and all(len(set(np.array([5, 5, 5, 7, 7])[b])) == len(b) for b in tiny)


def test_match_disjoint_evaluation_returns_predictions_in_row_order():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    groups = np.repeat(np.arange(50), 4)
    preds = []
    for g in (None, groups[100:]):
        torch.manual_seed(0)
        model = dtb.build_mlp(6, 8, 1, 0.0)  # rows are independent, so batching cannot matter
        out = dtb.train_torch(model, X[:100], y[:100], X[100:], y[100:], X[100:], y[100:],
                              epochs=1, batch_size=16, lr=1e-3, weight_decay=0.0, patience=1,
                              tokenized=False, label="order", eval_batch_size=16, device="cpu",
                              groups_va=g, groups_te=g)
        preds.append(out["_pred"])
    assert np.allclose(preds[0], preds[1], atol=1e-6)


# --------------------------------------------------------------------------- 3. attention / capacity
def _ft(**kw):
    return dtb.build_ft_transformer(13, 7, 16, 2, 4, 0.1, **kw)


def _saint(**kw):
    return dtb.build_saint(13, 7, 16, 2, 4, 0.1, d_misa=8, **kw)


ARCHS = [(_ft, "published"), (_ft, "gorishniy"), (_saint, "published"), (_saint, "paper")]


@pytest.mark.parametrize("builder,arch", ARCHS)
def test_sdpa_attention_matches_explicit_attention(builder, arch):
    torch.manual_seed(0)
    fast = builder(arch=arch, attn_impl="sdpa").eval()
    slow = builder(arch=arch, attn_impl="math").eval()
    slow.load_state_dict(fast.state_dict())
    x = torch.randn(9, 13, 7)
    with torch.no_grad():
        assert (fast(x) - slow(x)).abs().max().item() < 1e-4


def test_attend_matches_explicit_attention_at_the_real_token_count():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 8, 1017, 4, generator=g) for _ in range(3))
    diff = (dtb._attend(q, k, v, 0.0, "sdpa") - dtb._attend(q, k, v, 0.0, "math")).abs().max()
    assert diff.item() < 1e-4


@pytest.mark.parametrize("d_head", [4, 12, 113])
def test_padded_sdpa_is_exact_for_head_dims_not_divisible_by_8(d_head):
    g = torch.Generator().manual_seed(d_head)
    q, k, v = (torch.randn(2, 8, 50, d_head, generator=g) for _ in range(3))
    padded = dtb._attend(q, k, v, 0.0, "sdpa_pad")
    assert padded.shape == q.shape
    assert (padded - dtb._attend(q, k, v, 0.0, "math")).abs().max().item() < 1e-5


@pytest.mark.skipif(not CUDA, reason="no CUDA device")
def test_bf16_attention_at_head_dim_4_does_not_materialise_the_token_matrix():
    # d_token 32 / 8 heads under bf16: unpadded, SDPA falls back to the math kernel and allocates
    # the (64, 8, 1017, 1017) weights, 1.97 GiB
    q, k, v = (torch.randn(64, 8, 1017, 4, device="cuda", requires_grad=True) for _ in range(3))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = dtb._attend(*(t.to(torch.bfloat16) for t in (q, k, v)), 0.1, "sdpa")
    out.float().sum().backward()
    torch.cuda.synchronize()
    peak_gib = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    del q, k, v, out
    torch.cuda.empty_cache()
    assert peak_gib < 0.5


@pytest.mark.skipif(not CUDA, reason="no CUDA device")
@pytest.mark.parametrize("kind,arch", [("ft", "published"), ("ft", "gorishniy"), ("saint", "paper")])
def test_sdpa_matches_explicit_attention_on_cuda_at_1017_tokens(kind, arch):
    torch.manual_seed(0)
    if kind == "ft":
        make = lambda impl: dtb.build_ft_transformer(1016, 7, 128, 2, 8, 0.1, arch=arch,  # noqa: E731
                                                     attn_impl=impl)
    else:
        make = lambda impl: dtb.build_saint(1016, 7, 128, 2, 8, 0.1, arch=arch,  # noqa: E731
                                            attn_impl=impl)
    fast, slow = make("sdpa").cuda().eval(), make("math").cuda().eval()
    slow.load_state_dict(fast.state_dict())
    x = torch.randn(4, 1016, 7, device="cuda")
    with torch.no_grad():
        assert (fast(x) - slow(x)).abs().max().item() < 1e-4
    del fast, slow
    torch.cuda.empty_cache()


@pytest.mark.parametrize("builder,arch", ARCHS)
def test_gradient_checkpointing_is_exact(builder, arch):
    torch.manual_seed(0)
    plain = builder(arch=arch, grad_checkpoint=False)
    ckpt = builder(arch=arch, grad_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randn(6, 13, 7)
    results = []
    for model in (plain, ckpt):
        model.train()
        torch.manual_seed(123)  # identical dropout masks; checkpointing replays the RNG state
        out = model(x)
        out.sum().backward()
        results.append((out.detach(), [p.grad.clone() for p in model.parameters()]))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-6)
    for g_plain, g_ckpt in zip(results[0][1], results[1][1]):
        assert torch.allclose(g_plain, g_ckpt, atol=1e-6)


# --------------------------------------------------------------------------- 4. optimisation
def _lr_trace(schedule: str, steps: int = 100, lr: float = 1e-2):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=lr)
    scheduler = dtb.make_lr_scheduler(opt, schedule, lr, steps, warmup_frac=0.1,
                                      decay_rate=0.5, decay_steps=10)
    trace = []
    for _ in range(steps):
        trace.append(opt.param_groups[0]["lr"])
        param.grad = torch.ones(1)
        opt.step()
        if scheduler is not None:
            scheduler.step()
    return np.array(trace), scheduler


def test_lr_schedules():
    constant, scheduler = _lr_trace("constant")
    assert scheduler is None and np.allclose(constant, 1e-2)
    cosine, _ = _lr_trace("cosine_warmup")
    assert math.isclose(cosine[0], 1e-3) and math.isclose(cosine.max(), 1e-2)
    assert int(np.argmax(cosine)) in (9, 10) and np.all(np.diff(cosine[10:]) <= 1e-12)
    assert cosine[-1] < 1e-4
    onecycle, _ = _lr_trace("onecycle")
    assert math.isclose(onecycle[0], 1e-2 / 25) and 25 <= int(np.argmax(onecycle)) <= 31
    assert onecycle[-1] < 1e-5
    staircase, _ = _lr_trace("exp_decay")
    assert math.isclose(staircase[9], 1e-2) and math.isclose(staircase[10], 5e-3)
    assert math.isclose(staircase[25], 2.5e-3)
    with pytest.raises(ValueError):
        _lr_trace("linear")


def test_gorishniy_ft_decays_only_linear_weights():
    model = dtb.build_ft_transformer(5, 7, 16, 2, 4, 0.1, arch="gorishniy")
    groups = dtb._param_groups(model, 1e-5)
    linear_weights = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Linear)}
    assert {id(p) for p in groups[0]["params"]} == linear_weights
    assert groups[0]["weight_decay"] == 1e-5 and groups[1]["weight_decay"] == 0.0
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == len(list(model.parameters()))
    published = dtb.build_ft_transformer(5, 7, 16, 2, 4, 0.1)
    assert not isinstance(dtb._param_groups(published, 1e-5), list)  # one group, all decayed


# --------------------------------------------------------------------------- 5. CLI outputs
def test_cli_writes_per_model_predictions_in_test_row_order(tmp_path):
    shards = make_shards(tmp_path / "shards")
    output = tmp_path / "out" / "deep.json"
    models = ("lightgbm", "mlp", "tabnet", "ft_transformer", "saint")
    assert dtb.main([
        "--shards", str(shards), "--output", str(output), "--n-matches", "0", "--epochs", "1",
        "--models", ",".join(models), "--device", "cpu", "--n-boot", "10",
        "--d-token", "8", "--n-heads", "2", "--n-layers", "1", "--saint-d-misa", "8",
        "--mlp-hidden", "16", "--mlp-layers", "1", "--tabnet-n-d", "4", "--tabnet-n-steps", "2",
        "--saint-pretrain-epochs", "1", "--lgbm-threads", "1", "--batch-size", "32",
    ]) == 0
    rows = labelled_rows(shards)
    test = rows["patch"] == "15.16"
    results = json.loads(output.read_text(encoding="utf-8"))
    assert results["split"]["test"] == int(test.sum())
    assert results["label_key"] == "y_market_event" and "git_commit" in results
    assert results["deviations"]["protocol"]
    for name in models:
        path = tmp_path / "out" / f"deep.{name}.preds.npz"
        with np.load(path, allow_pickle=False) as blob:
            assert set(blob.files) == {"y", "pred", "groups"}
            assert np.array_equal(blob["y"], rows["y"][test])
            assert np.array_equal(blob["groups"], rows["groups"][test])
            assert blob["pred"].shape == (int(test.sum()),)
            assert np.isfinite(blob["pred"]).all()
        entry = results["models"][name]
        assert isinstance(entry["n_params"], int) and entry["n_params"] > 0
        assert entry["test_auc_ci95"] is not None
        assert results["deviations"][name]
    assert results["models"]["tabnet"]["sparsity_sign"] == "+"
    assert results["models"]["saint"]["pretrain"]["rows"] == int((rows["patch"] == "15.14").sum())
    with np.load(output.with_suffix(".preds.npz")) as legacy:  # the published combined file
        assert {"y_test", "groups_test", "min_participants_test"} <= set(legacy.files)
        assert {f"pred_{m}" for m in models} <= set(legacy.files)


def test_pair_preds_pairs_only_identical_test_rows(tmp_path):
    shards = make_shards(tmp_path / "shards")
    common = ["--shards", str(shards), "--n-matches", "0", "--epochs", "1", "--device", "cpu",
              "--n-boot", "20", "--mlp-hidden", "16", "--mlp-layers", "1", "--batch-size", "32"]
    first = tmp_path / "first" / "run.json"
    assert dtb.main([*common, "--output", str(first), "--models", "mlp"]) == 0
    assert "pairing" not in json.loads(first.read_text(encoding="utf-8"))  # schema unchanged
    other = make_shards(tmp_path / "other", seed=5)
    stranger = tmp_path / "stranger" / "run.json"
    assert dtb.main(["--shards", str(other), "--n-matches", "0", "--epochs", "1", "--device",
                     "cpu", "--n-boot", "0", "--mlp-hidden", "16", "--mlp-layers", "1",
                     "--output", str(stranger), "--models", "mlp"]) == 0
    second = tmp_path / "second" / "run.json"
    assert dtb.main([*common, "--output", str(second), "--models", "mlp,tabnet",
                     "--tabnet-n-d", "4", "--tabnet-n-steps", "2", "--pair-preds",
                     str(first.with_suffix(".preds.npz")), str(stranger.with_suffix(".preds.npz")),
                     str(tmp_path / "absent.preds.npz")]) == 0
    results = json.loads(second.read_text(encoding="utf-8"))
    notes = results["pairing"]
    assert any("paired (mlp) as published_mlp" in n for n in notes)
    assert any("test rows differ" in n for n in notes) and any("absent" in n for n in notes)
    boot = results["test_auc_bootstrap"]
    assert set(boot["paired"]) == {"mlp-published_mlp", "tabnet-published_mlp"}
    assert set(results["models"]) == {"mlp", "tabnet"}  # references are not models of this run
    rows = labelled_rows(shards)
    test = rows["patch"] == "15.16"
    refs, _ = dtb.load_reference_preds([tmp_path / "first" / "run.mlp.preds.npz"],
                                       rows["y"][test], rows["groups"][test])
    assert list(refs) == ["published_mlp"]  # the per-model file format pairs too
    refs, notes = dtb.load_reference_preds(
        [first.with_suffix(".preds.npz"), tmp_path / "first" / "run.mlp.preds.npz"],
        rows["y"][test], rows["groups"][test])
    assert list(refs) == ["published_mlp", "ref_run_mlp"] and "as ref_run_mlp" in notes[1]


def test_declared_grids_and_point_filter():
    grids = search.GRIDS["suggested"]
    sizes = {f: len(search.expand_grid(g)) for f, g in grids.items()}
    assert sizes == {"lightgbm": 12, "mlp": 24, "tabnet": 12, "ft_transformer": 36, "saint": 24,
                     "saint_pretrain": 24}
    last = search.expand_grid(grids["ft_transformer"])[-1]
    assert (last["d_token"], last["n_layers"], last["lr"]) == (192, 3, 1e-3)
    assert search.parse_points(None, 5) is None
    assert search.parse_points("0,-1", 5) == {0, 4}
    with pytest.raises(SystemExit):
        search.parse_points("5", 5)


def test_search_refuses_flags_that_do_not_apply(tmp_path):
    with pytest.raises(SystemExit, match="do not apply"):
        search.main(["--shards", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--split",
                     "match", "--dry-run"])


def test_search_selects_on_validation_and_scores_the_test_patch_once(tmp_path, monkeypatch):
    shards = make_shards(tmp_path / "shards")
    out = tmp_path / "search"
    calls = []
    real_fit = dtb.fit_model

    def spy(name, cfg, opts, X, y, tr, va, te, layout, label=None, groups=None):
        calls.append({"label": label, "tr": np.asarray(tr), "va": np.asarray(va),
                      "te": None if te is None else np.asarray(te), "groups": groups})
        return real_fit(name, cfg, opts, X, y, tr, va, te, layout, label=label, groups=groups)

    monkeypatch.setattr(dtb, "fit_model", spy)
    families = ("lightgbm", "tabnet", "saint_pretrain")
    argv = ["--shards", str(shards), "--out-dir", str(out), "--grid", "smoke",
            "--families", ",".join(families), "--pretrain-epochs", "1",
            "--search-train-matches", "10", "--n-boot", "10", "--published-preds",
            "--epochs", "1", "--device", "cpu", "--lgbm-threads", "1"]
    assert search.main(argv) == 0

    rows = labelled_rows(shards)
    patch, groups = rows["patch"], rows["groups"]
    searched = [c for c in calls if not c["label"].endswith(":refit")]
    refits = [c for c in calls if c["label"].endswith(":refit")]
    assert len(searched) == len(families) and len(refits) == len(families)
    for c in searched:
        assert c["te"] is None
        assert set(patch[c["tr"]]) == {"15.14"} and set(patch[c["va"]]) == {"15.15"}
        assert len(np.unique(groups[c["tr"]])) == 10
    for c in refits:
        assert set(patch[c["tr"]]) == {"15.14"} and len(c["tr"]) == int((patch == "15.14").sum())
        assert set(patch[c["va"]]) == {"15.15"} and len(c["va"]) == int((patch == "15.15").sum())
        assert set(patch[c["te"]]) == {"15.16"}

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    for family in families:
        record = summary["families"][family]
        assert record["n_done"] == record["n_declared_points"] == 1
        assert record["points"][0]["wall_clock_s"] is not None
        assert record["points"][0]["val_auc"] is not None
        assert record["refit"]["n_test_evaluations"] == 1
        assert record["refit"]["n_points_selected_from"] == 1
        assert record["refit"]["failed_point_indices"] == [] == record["refit"]["missing_point_indices"]
        assert record["refit"]["test_auc"] is not None and record["refit"]["test_auc_ci95"]
        with np.load(out / "refit" / f"{family}.preds.npz") as blob:
            assert set(blob.files) == {"y", "pred", "groups"}
            assert np.array_equal(blob["groups"], groups[patch == "15.16"])

    before = len(calls)
    assert search.main(argv) == 0  # resume: no grid point refitted, no test patch rescored
    assert len(calls) == before
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))[
        "points_run_this_invocation"] == 0
