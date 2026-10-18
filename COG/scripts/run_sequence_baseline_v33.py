"""BiGRU and temporal-Transformer baselines over the per-bin telemetry sequence (A3, CoG 2026 review R2).

R2 attributed the deep models' underperformance to the 6-step, 30-second observation window.  This
script fits two sequence encoders on the per-bin sequence before the cutoff for windows of 30, 60
and 120 s at 5-s bins (L = 6, 12, 24), on exactly the rows and the patch split of
scripts/run_window_sweep_v33.py, and pairs each against LightGBM fitted by that sweep to the same
rows.  Run the sweep first; this script reads its caches and never rebuilds features.

Input -- the sequence the pipeline produces
  gameplay/pipeline.build_ms_sequence builds L = ctx // BIN_MS bins ending at the engage cutoff; bin
  i covers [cutoff - ctx + i*BIN_MS, cutoff - ctx + (i+1)*BIN_MS): node and global snapshots read
  at the bin midpoint from frames strictly before the cutoff, events in the bin, absolute time_norm,
  causal anchors (preset v3.3).  gameplay/features.build_sequence_features turns it into x_seq
  (L x 1,015: role-ordered node features, per-player item hashes, global, event and spatial
  channels), which cfg.TEMPORAL_SEQ_PRIORITY = ("x_seq", "extra_seq") selects -- the tensor
  train/deep.py::pick_temporal_seq fed the CoG temporal models and the tensor seq_to_tabular
  summarises into the 7,105 telemetry columns LightGBM reads.  The sweep captures x_seq inside
  train.baseline.build_tabular_Xy (--save-seq).  frame_age_s (age of the last timeline frame at
  the cutoff, LightGBM's last column) is appended to every step as one more channel, so both
  learners see the same information.
  Before any fit, seq_to_tabular(x_seq) is recomputed for sampled rows and must equal the sweep's
  tabular matrix, and frame_age must equal its last column: the sequence is shown to be the input
  LightGBM summarised, under the same cutoff.  The sequence cache's signature must name the same
  fight index, match list, feature sources, commit and preset as the sweep summary, and the sweep's
  leak probe (post-cutoff rewrites leave x_seq bit-identical) must have passed.  The whole window
  precedes the cutoff, so reading it in both directions (BiGRU) or with unmasked self-attention
  (Transformer) observes nothing after the cutoff.

Learners (train/temporal_encoders.py)
  bigru        RNNEncoder("gru", bidirectional=True): gated recurrent unit (Cho et al., EMNLP 2014,
               sec. 2.3, eqs. 5-8) run in both directions (Schuster & Paliwal, IEEE TSP 45(11) 1997,
               sec. 3); readout = the final hidden states of both directions of the last layer.
  transformer  TransformerTemporalEncoder: multi-head self-attention encoder (Vaswani et al.,
               NeurIPS 2017, sec. 3.1-3.3) with sinusoidal positions (sec. 3.5), a learned [CLS]
               token (Devlin et al., NAACL 2019, sec. 3) and mean pooling as readout.
  Both feed a two-layer MLP head; binary cross-entropy, AdamW (Loshchilov & Hutter, ICLR 2019),
  gradient-norm clipping, early stopping on validation AUC with patience (Prechelt 1998, in
  Neural Networks: Tricks of the Trade, sec. 2.1), mixed precision with loss scaling on CUDA
  (Micikevicius et al., ICLR 2018).  Hyperparameters are fixed a priori (see --help).

Protocol and leak discipline
  train 15.14 / early stopping 15.15 / test 15.16 on the sweep's common rows.  Standardisation
  statistics come from the 15.14 rows only; the 15.15 AUC picks the epoch and nothing else; 15.16 is
  predicted once, with the selected weights.  Uncertainty: the sweep's match-clustered percentile
  bootstrap with ONE match draw per replicate shared by every prediction vector, so each paired
  delta resamples matches jointly (Efron & Tibshirani 1993, ch. 13; Field & Welsh 2007, JRSS-B 69(3),
  sec. 2.1).  Every deviation from the references is listed under "deviations" in the output JSON.

Resumability: each finished fit is cached as pred_<model>__<setting>__seed<k>.npz with a JSON
signature; an interrupted fit resumes from its per-epoch checkpoint.

    LOL_OUTPUT_ROOT=D:/LOL_Project LOL_CFG_PRESET=v3.3 python scripts/run_sequence_baseline_v33.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def _load_sweep_module():
    """scripts/run_window_sweep_v33.py imported by path (scripts/ is not a package)."""
    name = "run_window_sweep_v33"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SW = _load_sweep_module()
ITEM = SW.ITEM
PRESET = SW.PRESET
LABEL_KEY = SW.LABEL_KEY
SEED = SW.SEED
FRAME_AGE = SW.FRAME_AGE
MODELS = ("bigru", "transformer")
DEFAULT_CTX = "30,60,120"
BIN_MS = 5000
CLEAN_FLAGS = ("TIME_NORM_ABSOLUTE", "ANCHORS_CAUSAL", "TAB_FRAME_AGE_FEATURE")
REFERENCES = [
    "Cho et al. 2014, Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, EMNLP, sec. 2.3",
    "Schuster & Paliwal 1997, Bidirectional Recurrent Neural Networks, IEEE Trans. Signal Processing 45(11), sec. 3",
    "Vaswani et al. 2017, Attention Is All You Need, NeurIPS, sec. 3",
    "Devlin et al. 2019, BERT, NAACL, sec. 3 ([CLS] readout)",
    "Xiong et al. 2020, On Layer Normalization in the Transformer Architecture, ICML (pre-LN)",
    "Loshchilov & Hutter 2019, Decoupled Weight Decay Regularization, ICLR (AdamW)",
    "Prechelt 1998, Early Stopping - But When?, Neural Networks: Tricks of the Trade, sec. 2.1",
    "Micikevicius et al. 2018, Mixed Precision Training, ICLR (loss scaling)",
    "Efron & Tibshirani 1993, An Introduction to the Bootstrap, ch. 13",
    "Field & Welsh 2007, Bootstrapping clustered data, JRSS-B 69(3), sec. 2.1",
]
log = SW.log


class SweepMismatch(RuntimeError):
    """The sweep artefacts do not describe the rows, preset or inputs this baseline requires."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SweepMismatch(message)


# ------------------------------------------------------------------ sweep artefacts

def load_sweep(sweep_dir: Path, sweep_json: Path, patches: tuple[str, str, str], allow_missing_leak_probe: bool) -> dict:
    """Summary, common rows and LightGBM patch predictions of a finished window sweep, cross-checked."""
    summary = json.loads(Path(sweep_json).read_text(encoding="utf-8"))
    require(summary.get("item") == ITEM, f"{sweep_json} is not an {ITEM} summary")
    require((summary.get("preset"), summary.get("label_key"), summary.get("tie_policy")) == (PRESET, LABEL_KEY, "drop"),
            f"sweep ran {summary.get('preset')}/{summary.get('label_key')}/{summary.get('tie_policy')}, need {PRESET}/{LABEL_KEY}/drop")
    snap = summary.get("cfg", {})
    require(all(snap.get(k) is True for k in CLEAN_FLAGS) and int(snap.get("PREDICTION_GAP_MS", -1)) == 0,
            f"sweep was not run on the clean feature path: {({k: snap.get(k) for k in CLEAN_FLAGS})}")
    require(summary.get("inputs_changed", {}).get("passed") is True, "the sweep did not prove that its inputs changed")
    probe = summary.get("leak_probe", {}).get("passed")
    require(probe is True or (probe is None and allow_missing_leak_probe),
            f"the sweep's leak probe did not pass (passed={probe}); --allow-missing-leak-probe accepts a sweep run without it")
    split = summary.get("split", {})
    require((split.get("train"), split.get("val"), split.get("test")) == tuple(patches),
            f"sweep split {split.get('train')}/{split.get('val')}/{split.get('test')} differs from {patches}")
    with np.load(Path(sweep_dir) / "common_rows.npz", allow_pickle=False) as z:
        rows = {k: z[k] for k in z.files}
    require("ref_ordinal" in rows, "common_rows.npz lacks ref_ordinal: rerun the sweep with the current script")
    require(SW.common_rows_sha1(rows["match_id"], rows["engage_ts"], rows["y"]) == summary["common_rows"]["sha1"],
            "common_rows.npz does not match the sweep summary")
    preds_path = Path(summary["outputs"]["preds"])
    if not preds_path.exists():
        preds_path = Path(sweep_dir) / preds_path.name
    with np.load(preds_path, allow_pickle=False) as z:
        preds = {k: z[k] for k in z.files}
    for key in ("match_id", "engage_ts", "y", "patch"):
        require(np.array_equal(preds[key], rows[key]), f"{preds_path.name}: row order differs from common_rows.npz ({key})")
    test_index = np.flatnonzero(rows["patch"].astype(str) == patches[2])
    require(np.array_equal(preds["patch_test_index"], test_index), "the sweep's patch test rows differ from the common rows' test patch")
    return {"summary": summary, "rows": rows, "preds": preds, "preds_path": preds_path, "test_index": test_index}


def common_keys(rows: dict) -> list[tuple[str, int, int]]:
    return list(zip(np.asarray(rows["match_id"]).astype(str).tolist(), np.asarray(rows["engage_ts"]).astype(np.int64).tolist(),
                    np.asarray(rows["ref_ordinal"]).astype(np.int64).tolist()))


def load_sequences(sweep_dir: Path, name: str, sweep: dict, ctx_sec: int, bin_ms: int) -> dict:
    """The sweep's x_seq cache for one setting, restricted to the common rows in common-row order."""
    path = Path(sweep_dir) / f"seq_{name}.npz"
    side = SW.sidecar(path)
    require(path.exists() and side.exists(), f"missing {path}: run the sweep with --save-seq including {ctx_sec}:{bin_ms}")
    sig = json.loads(side.read_text(encoding="utf-8"))["signature"]
    s = sweep["summary"]
    expect = {"kind": "sequence", "setting": name, "preset": PRESET, "label": LABEL_KEY, "tie_policy": "drop", "seq_key": "x_seq",
              "ctx_sec": int(ctx_sec), "bin_ms": int(bin_ms), "index_sha1": s["fight_index"]["sha1"],
              "match_list_sha1": s["match_list_sha1"], "source_sha1": s["source_sha1"], "script_sha1": s["script_sha1"],
              "feature_set": s["feature_set"]}
    bad = {k: [sig.get(k), v] for k, v in expect.items() if sig.get(k) != v}
    require(not bad, f"{path.name} was not written by this sweep run: {bad}")
    require(sig.get("preset_values") == s["preset_values"] and all(sig["preset_values"].get(k) is True for k in CLEAN_FLAGS),
            f"{path.name}: preset values differ from the sweep's or the clean feature path is off")
    rows = sweep["rows"]
    with np.load(path, allow_pickle=False) as z:
        keys = list(zip(z["match_id"].astype(str).tolist(), z["engage_ts"].astype(np.int64).tolist(),
                        z["ref_ordinal"].astype(np.int64).tolist()))
        pos = {k: i for i, k in enumerate(keys)}
        require(len(pos) == len(keys), f"{path.name}: duplicate row keys")
        wanted = common_keys(rows)
        missing = sum(1 for k in wanted if k not in pos)
        require(missing == 0, f"{missing} common rows are absent from {path.name}")
        take = np.fromiter((pos[k] for k in wanted), dtype=np.int64, count=len(wanted))
        require(np.array_equal(z["y"][take].astype(np.int8), rows["y"].astype(np.int8)), f"{path.name}: labels differ from the common rows")
        require(np.array_equal(z["patch"][take].astype(str), rows["patch"].astype(str)), f"{path.name}: patches differ from the common rows")
        seq = z["seq"]
        subset_or_reordered = not (len(take) == len(seq) and np.array_equal(take, np.arange(len(take))))
        if subset_or_reordered:
            seq = seq[take]
        frame_age = z["frame_age"][take].astype(np.float32)
        seq_names = z["seq_names"].astype(str).tolist()
    require(seq.ndim == 3 and seq.shape[1] == SW.n_bins(ctx_sec, bin_ms) and seq.shape[2] == len(seq_names),
            f"{path.name}: shape {seq.shape} does not match L={SW.n_bins(ctx_sec, bin_ms)}, D={len(seq_names)}")
    return {"seq": seq, "frame_age": frame_age, "names": seq_names, "signature": sig, "path": path,
            "L": int(seq.shape[1]), "D": int(seq.shape[2]), "n_rows_cached": int(len(keys)),
            "subset_or_reordered": bool(subset_or_reordered)}


def verify_against_tabular(sweep_dir: Path, name: str, seqd: dict, rows: dict, n_check: int, seed: int) -> dict:
    """seq_to_tabular(x_seq) must reproduce the sweep's tabular matrix on sampled common rows."""
    from gameplay.features import seq_to_tabular
    path = Path(sweep_dir) / f"setting_{name}.npz"
    with np.load(path, allow_pickle=False) as z:
        names = z["names"].astype(str).tolist()
        keys = list(zip(z["row_match_id"].astype(str).tolist(), z["row_t_start_ts"].astype(np.int64).tolist(),
                        z["row_ref_ordinal"].astype(np.int64).tolist()))
        X = z["X"]
    require(names[-1] == FRAME_AGE and len(names) == 7 * seqd["D"] + 1,
            f"{path.name}: {len(names)} columns, expected 7 x {seqd['D']} summaries + {FRAME_AGE}")
    pos = {k: i for i, k in enumerate(keys)}
    wanted = common_keys(rows)
    rng = np.random.default_rng(int(seed))
    pick = np.sort(rng.choice(len(wanted), size=min(int(n_check), len(wanted)), replace=False))
    exact = close = 0
    worst = 0.0
    failures = []
    for i in pick.tolist():
        x = X[pos[wanted[i]]]
        tab = seq_to_tabular(seqd["seq"][i])
        fa_ok = bool(np.float32(seqd["frame_age"][i]) == x[-1])
        if np.array_equal(tab, x[:-1], equal_nan=True) and fa_ok:
            exact += 1
        if np.allclose(tab, x[:-1], rtol=1e-5, atol=1e-5, equal_nan=True) and fa_ok:
            close += 1
        else:
            failures.append(list(wanted[i]))
        finite = np.isfinite(tab) & np.isfinite(x[:-1])
        if finite.any():
            worst = max(worst, float(np.max(np.abs(tab[finite].astype(np.float64) - x[:-1][finite].astype(np.float64)))))
    del X
    out = {"setting_cache": str(path), "rows_checked": int(len(pick)), "rows_bit_identical": int(exact),
           "rows_within_1e-5": int(close), "max_abs_difference": worst, "failures": failures[:3]}
    require(close == len(pick), f"x_seq is not the input the tabular matrix summarises: {out}")
    return out


# ------------------------------------------------------------------ standardisation (training rows only)

def fit_standardiser(seq: np.ndarray, frame_age: np.ndarray, rows: np.ndarray, chunk: int = 1024) -> dict:
    """Per-channel mean and SD over the given rows and all time steps; frame_age is channel D.

    Non-finite inputs count as 0 (their number is returned).  Only `rows` are read, so passing the
    training-patch rows keeps validation and test values out of the statistics."""
    rows = np.asarray(rows, dtype=np.int64)
    L, D = int(seq.shape[1]), int(seq.shape[2])
    s1 = np.zeros(D + 1, dtype=np.float64)
    s2 = np.zeros(D + 1, dtype=np.float64)
    nonfinite = 0
    for s in range(0, len(rows), chunk):
        idx = rows[s:s + chunk]
        part = seq[idx].astype(np.float64)
        nonfinite += int((~np.isfinite(part)).sum())
        part = np.nan_to_num(part, nan=0.0, posinf=0.0, neginf=0.0)
        s1[:D] += part.sum(axis=(0, 1))
        s2[:D] += np.square(part).sum(axis=(0, 1))
        fa = np.nan_to_num(np.asarray(frame_age, dtype=np.float64)[idx], nan=0.0, posinf=0.0, neginf=0.0)
        s1[D] += fa.sum() * L
        s2[D] += np.square(fa).sum() * L
    count = float(len(rows) * L)
    if count <= 0:
        raise ValueError("no training rows for the standardiser")
    mu = s1 / count
    sd = np.sqrt(np.maximum(s2 / count - mu ** 2, 0.0))
    constant = sd < 1e-6
    sd[constant] = 1.0
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "n_rows": int(len(rows)),
            "n_constant_channels": int(constant.sum()), "n_nonfinite": int(nonfinite)}


def standardise(seq: np.ndarray, frame_age: np.ndarray, stats: dict, clip: float, chunk: int = 1024) -> tuple[np.ndarray, int]:
    """(N, L, D) sequence + (N,) frame age -> (N, L, D+1) float32 z-scores clipped to +-clip."""
    N, L, D = (int(v) for v in seq.shape)
    mu, sd = stats["mu"], stats["sd"]
    out = np.empty((N, L, D + 1), dtype=np.float32)
    nonfinite = 0
    for s in range(0, N, chunk):
        part = np.asarray(seq[s:s + chunk], dtype=np.float32)
        bad = ~np.isfinite(part)
        nonfinite += int(bad.sum())
        if bad.any():
            part = np.where(bad, np.float32(0.0), part)
        blk = out[s:s + chunk]
        np.subtract(part, mu[:D], out=blk[:, :, :D])
        np.divide(blk[:, :, :D], sd[:D], out=blk[:, :, :D])
        fa = np.nan_to_num(np.asarray(frame_age[s:s + chunk], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        blk[:, :, D] = ((fa - mu[D]) / sd[D])[:, None]
        np.clip(blk, -float(clip), float(clip), out=blk)
    return out, nonfinite


def share_at_clip(X: np.ndarray, clip: float, chunk: int = 1024) -> float:
    """Share of standardised cells sitting at +-clip, chunked (np.abs(X) over the whole array would copy it)."""
    hits = 0
    for s in range(0, len(X), chunk):
        hits += int(np.count_nonzero(np.abs(X[s:s + chunk]) >= float(clip)))
    return hits / max(1, int(X.size))


# ------------------------------------------------------------------ models

class SequenceClassifier(nn.Module):
    """A temporal encoder from train/temporal_encoders.py and a two-layer MLP logit head."""

    def __init__(self, kind: str, d_in: int, seq_len: int, hp: dict):
        super().__init__()
        from train.temporal_encoders import RNNEncoder, TransformerTemporalEncoder
        self.kind = str(kind)
        self.readout = hp.get("readout")
        self.grad_checkpoint = bool(hp.get("grad_checkpoint", False))
        if self.kind == "bigru":
            self.encoder = RNNEncoder("gru", int(d_in), int(hp["hidden"]), n_layers=int(hp["layers"]),
                                      bidirectional=True, dropout=float(hp["dropout"]))
            if self.encoder.attn_pool is not None:
                raise ValueError("cfg.USE_ATTENTION_POOL must be False: the BiGRU readout is the final states")
            out_dim = 2 * int(hp["hidden"])
        elif self.kind == "transformer":
            if int(hp["d_model"]) % int(hp["nhead"]):
                raise ValueError("d_model must be divisible by nhead")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
                self.encoder = TransformerTemporalEncoder(int(d_in), int(hp["d_model"]), int(hp["nhead"]), int(hp["layers"]),
                                                          dropout=float(hp["dropout"]), max_len=int(seq_len))
            out_dim = int(self.encoder.out_dim)
        else:
            raise ValueError(f"unknown model {kind!r}; known: {MODELS}")
        self.head = nn.Sequential(nn.Linear(out_dim, int(hp["head_hidden"])), nn.ReLU(), nn.Dropout(float(hp["head_dropout"])),
                                  nn.Linear(int(hp["head_hidden"]), 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "bigru" and self.readout == "final_states":
            _, h_n = self.encoder.rnn(x)                      # (layers x 2, B, H): ..., last fwd, last bwd
            return torch.cat([h_n[-2], h_n[-1]], dim=-1)      # forward after bin L-1, backward after bin 0
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            z = checkpoint(self.encode, x, use_reentrant=False)
        else:
            z = self.encode(x)
        return self.head(z).squeeze(-1)


def hyperparameters(args, kind: str) -> dict:
    from core.config import cfg
    hp = {"batch_size": int(args.batch_size), "eval_batch_size": int(args.eval_batch_size), "lr": float(args.lr),
          "weight_decay": float(args.weight_decay), "max_epochs": int(args.epochs), "patience": int(args.patience),
          "min_delta": float(args.min_delta), "grad_clip": float(args.grad_clip), "head_hidden": int(args.head_hidden),
          "head_dropout": float(args.dropout), "clip_z": float(args.clip), "amp": str(args.amp),
          "grad_checkpoint": bool(args.grad_checkpoint), "frame_age_channel": True, "optimizer": "AdamW",
          "loss": "binary cross-entropy", "early_stopping_metric": "validation-patch AUC"}
    if kind == "bigru":
        hp.update(hidden=int(args.gru_hidden), layers=int(args.gru_layers), dropout=float(args.dropout), bidirectional=True,
                  readout=str(args.bigru_readout))
    else:
        hp.update(d_model=int(args.d_model), nhead=int(args.nhead), layers=int(args.trf_layers), dropout=float(args.trf_dropout),
                  ff_mult=int(getattr(cfg, "TRANS_FF_MULT", 2)), norm_first=True, positional="sinusoidal", readout="cls+mean")
    return hp


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, rows: np.ndarray, *, batch: int, device, amp_dtype) -> np.ndarray:
    model.eval()
    use_amp = device.type == "cuda" and amp_dtype is not None
    out = np.empty(len(rows), dtype=np.float64)
    for s in range(0, len(rows), int(batch)):
        sel = rows[s:s + int(batch)]
        xb = torch.from_numpy(X[sel]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
            logits = model(xb)
        out[s:s + len(sel)] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def _cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def fit_sequence_model(kind: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray, hp: dict, *,
                       seed: int, device, amp_dtype, ckpt_path: Path | None, ckpt_signature: dict) -> dict:
    """Fit on `tr`, pick the epoch by AUC on `va`, predict `va` and `te` once with the picked weights."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device)
    L, d_in = int(X.shape[1]), int(X.shape[2])
    model = SequenceClassifier(kind, d_in, L, hp).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    use_amp = cuda and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = {"epoch": 0, "best_auc": float("-inf"), "best_epoch": -1, "bad_epochs": 0, "history": [], "seconds": 0.0}
    best_state = None
    resumed_from = None
    if ckpt_path is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("signature") == ckpt_signature:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            state, best_state = ck["state"], ck["best_state"]
            torch.set_rng_state(ck["torch_rng"])
            if cuda and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"], device)
            resumed_from = int(state["epoch"])
            log(f"      resumed from checkpoint after epoch {resumed_from}")
    y32 = np.asarray(y, dtype=np.float32)
    while state["epoch"] < hp["max_epochs"] and state["bad_epochs"] < hp["patience"]:
        epoch = int(state["epoch"])
        t0 = time.time()
        model.train()
        order = np.random.default_rng([int(seed), epoch]).permutation(tr)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for s in range(0, len(order), hp["batch_size"]):
            idx = np.sort(order[s:s + hp["batch_size"]])
            xb = torch.from_numpy(X[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(y32[idx]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.detach().double() * len(idx)
        val_auc = float(roc_auc_score(y[va], predict(model, X, va, batch=hp["eval_batch_size"], device=device, amp_dtype=amp_dtype)))
        improved = val_auc > state["best_auc"] + hp["min_delta"]
        if improved:
            state.update(best_auc=val_auc, best_epoch=epoch + 1, bad_epochs=0)
            best_state = _cpu_state(model)
        else:
            state["bad_epochs"] += 1
        seconds = round(time.time() - t0, 2)
        state["seconds"] += seconds
        state["history"].append({"epoch": epoch + 1, "train_loss": float(loss_sum.item() / max(1, len(order))),
                                 "val_auc": val_auc, "seconds": seconds, "loss_scale": float(scaler.get_scale()) if use_amp else None})
        state["epoch"] = epoch + 1
        log(f"      epoch {epoch + 1}/{hp['max_epochs']} loss {state['history'][-1]['train_loss']:.4f} "
            f"val-auc {val_auc:.4f}{' *' if improved else ''} ({seconds}s)")
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path.with_name(ckpt_path.name + ".partial")
            torch.save({"signature": ckpt_signature, "model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "state": state, "best_state": best_state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if cuda else None}, tmp)
            os.replace(tmp, ckpt_path)
    if best_state is None:
        raise RuntimeError(f"{kind}: no epoch was trained")
    model.load_state_dict(best_state)
    pred_val = predict(model, X, va, batch=hp["eval_batch_size"], device=device, amp_dtype=amp_dtype)
    pred_test = predict(model, X, te, batch=hp["eval_batch_size"], device=device, amp_dtype=amp_dtype)
    fit = {"model": kind, "n_params": n_params, "epochs_run": int(state["epoch"]), "best_epoch": int(state["best_epoch"]),
           "val_auc_best_epoch": float(state["best_auc"]), "val_auc_reloaded": float(roc_auc_score(y[va], pred_val)),
           "stopped_by_patience": bool(state["bad_epochs"] >= hp["patience"]), "resumed_from_epoch": resumed_from,
           "history": state["history"], "train_seconds": round(float(state["seconds"]), 1),
           "test_auc": float(roc_auc_score(y[te], pred_test)),
           "cuda_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1) if cuda else None}
    return {"fit": fit, "pred_val": pred_val, "pred_test": pred_test}


# ------------------------------------------------------------------ main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=SW.DEFAULT_OUT_DIR, help="output directory of run_window_sweep_v33.py")
    ap.add_argument("--sweep-json", type=Path, default=None, help="default <sweep-dir>/window_sweep_v33.json")
    ap.add_argument("--ctx", default=DEFAULT_CTX, help="comma list of window lengths in seconds")
    ap.add_argument("--bin-ms", type=int, default=BIN_MS)
    ap.add_argument("--reference-ctx", type=int, default=30, help="window every other window is paired against")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=str(SEED), help="comma list; each seed is a separate fit and prediction vector")
    ap.add_argument("--train-patch", default="15.14")
    ap.add_argument("--val-patch", default="15.15")
    ap.add_argument("--test-patch", default="15.16")
    ap.add_argument("--epochs", type=int, default=40, help="maximum epochs")
    ap.add_argument("--patience", type=int, default=6, help="epochs without a validation-AUC gain before stopping")
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--gru-hidden", type=int, default=128)
    ap.add_argument("--gru-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2, help="between GRU layers and in both heads")
    ap.add_argument("--bigru-readout", choices=("final_states", "encoder"), default="final_states",
                    help="final_states: last-layer h_n of both directions; encoder: RNNEncoder.forward (out[:, -1], the CoG pipeline)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--trf-layers", type=int, default=2)
    ap.add_argument("--trf-dropout", type=float, default=0.1)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--clip", type=float, default=10.0, help="clip standardised inputs to +-clip")
    ap.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16", help="CUDA autocast dtype")
    ap.add_argument("--grad-checkpoint", action="store_true", help="activation checkpointing of the encoder (memory for compute)")
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu; cuda must be available unless cpu is named")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--verify-tabular-rows", type=int, default=512, help="rows whose seq_to_tabular(x_seq) is checked (0 = skip)")
    ap.add_argument("--allow-missing-leak-probe", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="refit even when a cached prediction or checkpoint exists")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <sweep-dir>/sequence_baseline")
    ap.add_argument("--output", type=Path, default=None, help="default <out-dir>/sequence_baseline_v33.json")
    return ap.parse_args(argv)


def deviations(args) -> list[str]:
    out = [
        "torch.nn.GRU (cuDNN) applies the reset gate after the recurrent linear map, "
        "n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn)); Cho et al. 2014 eq. 8 applies it before it, "
        "tanh(W x + U (r * h_{t-1})).  This is PyTorch's documented GRU variant.",
        "Transformer: pre-layer normalisation (norm_first=True, Xiong et al. 2020) instead of the post-LN of Vaswani et al. "
        "2017, and without the final LayerNorm Xiong et al. place after the pre-LN stack (nn.TransformerEncoder is built with "
        "norm=None); feed-forward width cfg.TRANS_FF_MULT x d_model (2x; the paper uses 4x); the input is projected by a "
        "linear map without the sqrt(d_model) scaling of sec. 3.4, and no dropout is applied to the sum of projection and "
        "positional encoding (sec. 5.4 applies it); readout is a learned [CLS] token concatenated with mean pooling over the "
        "L bins (Vaswani et al. define no sequence-classification readout).",
        "Hyperparameters are fixed a priori (recorded per fit), not tuned; the only selection on 15.15 is the early-stopping epoch.",
        "frame_age_s is appended to every time step as a constant channel (LightGBM receives it as one column).",
        "Inputs are z-scored with 15.14 statistics pooled over time steps, clipped to +-clip; non-finite values are set to 0 "
        "before standardisation (counts recorded).",
        "Rows are the window sweep's --n-matches sample (default 20,000 matches), not all 191,940 matches; the LightGBM "
        "comparator is the sweep's lgbm_paper patch fit on the same rows, whose column filter uses 15.14 rows only.",
        "cuDNN recurrent kernels are not bit-deterministic; a rerun with the same seed can differ in late digits.",
        "Bootstrap AUCs are multiplicity-weighted Mann-Whitney statistics (numerically identical to concatenating resampled rows).",
    ]
    if args.bigru_readout == "final_states":
        out.append("BiGRU readout uses h_n of both directions of the last layer, not RNNEncoder.forward's out[:, -1], whose backward "
                   "half has read only the last bin; --bigru-readout encoder reproduces the CoG pipeline's readout.")
    else:
        out.append("BiGRU readout is RNNEncoder.forward's out[:, -1] (the CoG pipeline): its backward half has read only the last bin, "
                   "unlike the final-state readout of Schuster & Paliwal 1997.")
    if args.amp != "off" and args.device != "cpu":
        out.append(f"Mixed precision ({args.amp} autocast) with a gradient scaler.  The scaler stays on under bfloat16 because "
                   "cuDNN runs nn.GRU in float16 even under bfloat16 autocast (torch 2.11/cuDNN 9.10: output dtype float16), "
                   "so the recurrent gradients still need loss scaling (Micikevicius et al. 2018, sec. 3.2).")
    if args.grad_checkpoint:
        out.append("Activation checkpointing of the encoder (non-reentrant, RNG state preserved) trades compute for memory; "
                   "the function computed is unchanged.")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu to run on the CPU deliberately")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp] if device.type == "cuda" else None
    torch.backends.cudnn.benchmark = False
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models or not set(models) <= set(MODELS):
        raise SystemExit(f"--models must be a subset of {MODELS}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ctxs = sorted({int(c) for c in args.ctx.split(",") if c.strip()})
    names = [SW.setting_name(c, args.bin_ms) for c in ctxs]
    ref_name = SW.setting_name(args.reference_ctx, args.bin_ms)
    sweep_dir = Path(args.sweep_dir)
    sweep_json = args.sweep_json or sweep_dir / "window_sweep_v33.json"
    out_dir = Path(args.out_dir or sweep_dir / "sequence_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output or out_dir / "sequence_baseline_v33.json")
    patches = (args.train_patch, args.val_patch, args.test_patch)

    sweep = load_sweep(sweep_dir, sweep_json, patches, args.allow_missing_leak_probe)
    s = sweep["summary"]
    rows = sweep["rows"]
    y = rows["y"].astype(np.int8)
    groups = rows["match_id"].astype(str)
    patch = rows["patch"].astype(str)
    tr, va, te = (np.flatnonzero(patch == p) for p in patches)
    for label, idx in (("train", tr), ("val", va), ("test", te)):
        require(len(idx) > 0 and len(np.unique(y[idx])) == 2, f"{label} patch needs rows of both classes")
    split = {"kind": "patch holdout", "train": args.train_patch, "val": args.val_patch, "test": args.test_patch,
             "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
             "matches": {k: int(len(set(groups[i].tolist()))) for k, i in (("train", tr), ("val", va), ("test", te))},
             "use": {"train": "fit and standardisation statistics", "val": "early-stopping epoch only", "test": "scored once"}}
    for name in names:
        require(name in s["settings"], f"{name} is not a setting of the sweep ({sorted(s['settings'])})")
        require(f"patch__pred__{name}" in sweep["preds"], f"the sweep has no LightGBM patch predictions for {name}")
    git = SW.git_info()
    script_sha1 = SW.file_sha1(Path(__file__))
    from core.config import cfg
    encoder_cfg = {"TRANS_FF_MULT": int(getattr(cfg, "TRANS_FF_MULT", 2)), "USE_ATTENTION_POOL": bool(getattr(cfg, "USE_ATTENTION_POOL", False))}
    require(not encoder_cfg["USE_ATTENTION_POOL"], "cfg.USE_ATTENTION_POOL is on; this baseline defines its own readouts")
    cuda_info = None
    if device.type == "cuda":
        cuda_info = {"device": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(), "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                     "sdpa_flash": bool(torch.backends.cuda.flash_sdp_enabled()),
                     "sdpa_mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled())}
    log(f"[{ITEM}/sequence] sweep {sweep_json} ({s['common_rows']['n_rows']} common rows, {s['common_rows']['n_matches']} matches); "
        f"split rows {split['rows']}; device {device} amp {args.amp if amp_dtype is not None else 'off'}")

    test_preds = {f"lgbm__{name}": sweep["preds"][f"patch__pred__{name}"].astype(np.float64) for name in names}
    val_preds: dict = {}
    fits: dict = {}
    inputs: dict = {}
    for ctx_sec, name in zip(ctxs, names):
        t0 = time.time()
        seqd = load_sequences(sweep_dir, name, sweep, ctx_sec, args.bin_ms)
        check = verify_against_tabular(sweep_dir, name, seqd, rows, args.verify_tabular_rows, SEED) if args.verify_tabular_rows > 0 else None
        stats = fit_standardiser(seqd["seq"], seqd["frame_age"], tr)
        X, nonfinite_all = standardise(seqd["seq"], seqd["frame_age"], stats, args.clip)
        inputs[name] = {"ctx_sec": ctx_sec, "bin_ms": args.bin_ms, "L": seqd["L"], "D_seq": seqd["D"], "d_in": int(X.shape[2]),
                        "seq_key": "x_seq", "sequence_cache": str(seqd["path"]), "sequence_signature": seqd["signature"],
                        "tabular_consistency": check,
                        "standardiser": {"rows": "train patch only", "n_rows": stats["n_rows"], "n_constant_channels": stats["n_constant_channels"],
                                         "n_nonfinite_train": stats["n_nonfinite"], "n_nonfinite_all_rows": int(nonfinite_all),
                                         "share_clipped": share_at_clip(X, args.clip)},
                        "load_seconds": round(time.time() - t0, 1)}
        log(f"[{name}] X {X.shape} (L={seqd['L']}), tabular check {check['rows_within_1e-5'] if check else 'skipped'}/"
            f"{check['rows_checked'] if check else 0}, clipped share {inputs[name]['standardiser']['share_clipped']:.2e}")
        seq_signature = seqd["signature"]
        del seqd
        for kind in models:
            hp = hyperparameters(args, kind)
            for seed in seeds:
                key = f"{kind}__{name}__seed{seed}"
                cache = out_dir / f"pred_{key}.npz"
                sig = {"item": ITEM, "kind": "sequence_pred", "model": kind, "setting": name, "seed": int(seed), "hp": hp,
                       "sequence_signature": seq_signature, "common_rows_sha1": s["common_rows"]["sha1"], "split": split,
                       "script_sha1": script_sha1, "device": device.type, "encoder_cfg": encoder_cfg}
                ckpt = out_dir / "checkpoints" / f"ckpt_{key}.pt"
                if not args.no_cache and SW.cache_valid(cache, sig):
                    with np.load(cache, allow_pickle=False) as z:
                        test_preds[key], val_preds[key] = z["pred_test"], z["pred_val"]
                    fits[key] = {**json.loads(SW.sidecar(cache).read_text(encoding="utf-8"))["fit"], "from_cache": True}
                    log(f"[{name}] {kind} seed {seed}: cached (test AUC {fits[key]['test_auc']:.4f})")
                    continue
                if args.no_cache and ckpt.exists():
                    ckpt.unlink()
                log(f"[{name}] {kind} seed {seed}: fitting")
                result = fit_sequence_model(kind, X, y, tr, va, te, hp, seed=seed, device=device, amp_dtype=amp_dtype,
                                            ckpt_path=ckpt, ckpt_signature=SW.canonical(sig))
                fit = {**result["fit"], "setting": name, "seed": int(seed), "hp": hp, "from_cache": False}
                test_preds[key], val_preds[key] = result["pred_test"], result["pred_val"]
                SW.atomic_savez(cache, pred_test=result["pred_test"], pred_val=result["pred_val"])
                SW.write_json(SW.sidecar(cache), {"signature": SW.canonical(sig), "fit": fit})
                if ckpt.exists():
                    ckpt.unlink()
                fits[key] = fit
                log(f"[{name}] {kind} seed {seed}: best epoch {fit['best_epoch']}/{fit['epochs_run']} val {fit['val_auc_best_epoch']:.4f} "
                    f"test {fit['test_auc']:.4f}, {fit['n_params']} params, cuda peak {fit['cuda_max_memory_allocated_mb']} MB")
        del X
    try:
        (out_dir / "checkpoints").rmdir()             # only succeeds when every fit finished and removed its checkpoint
    except OSError:
        pass

    pairs = []
    for name in names:
        for seed in seeds:
            for kind in models:
                key = f"{kind}__{name}__seed{seed}"
                pairs.append((key, f"lgbm__{name}"))                       # same rows, same window
                if name != ref_name and ref_name in names:
                    pairs.append((key, f"lgbm__{ref_name}"))               # longer window vs LightGBM at the published 30 s
                    pairs.append((key, f"{kind}__{ref_name}__seed{seed}"))
            if set(MODELS) <= set(models):
                pairs.append((f"transformer__{name}__seed{seed}", f"bigru__{name}__seed{seed}"))
        if name != ref_name and ref_name in names:
            pairs.append((f"lgbm__{name}", f"lgbm__{ref_name}"))
    boot = SW.match_bootstrap(y[te], test_preds, groups[te], n_boot=args.n_boot, seed=SEED, pairs=pairs)
    for k, v in boot["auc"].items():
        log(f"[test] {k}: AUC {v['auc']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
    for k, v in boot["paired"].items():
        log(f"[test] {k}: {v['delta']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")

    peak = [f["cuda_max_memory_allocated_mb"] for f in fits.values() if f.get("cuda_max_memory_allocated_mb") is not None]
    preds_out = output.with_suffix(".preds.npz")
    store = {"match_id": groups[te], "engage_ts": rows["engage_ts"][te], "ref_ordinal": rows["ref_ordinal"][te], "y": y[te],
             "test_index": te, "val_index": va}
    store.update({f"test__{k}": v for k, v in test_preds.items()})
    store.update({f"val__{k}": v for k, v in val_preds.items()})
    summary = {
        "item": ITEM, "script": "scripts/run_sequence_baseline_v33.py", "script_sha1": script_sha1, "git": git,
        "preset": PRESET, "preset_values": s["preset_values"], "label_key": LABEL_KEY, "tie_policy": "drop",
        "seeds": seeds, "bootstrap_seed": SEED,
        "sweep": {"json": str(sweep_json), "git_commit": s["git"]["commit"], "source_sha1": s["source_sha1"],
                  "fight_index_sha1": s["fight_index"]["sha1"], "match_list_sha1": s["match_list_sha1"],
                  "n_matches_sampled": s["n_matches_sampled"], "common_rows_sha1": s["common_rows"]["sha1"],
                  "leak_probe_passed": s.get("leak_probe", {}).get("passed"), "inputs_changed_passed": s["inputs_changed"]["passed"],
                  "lgbm_patch_fits": {n: s["protocols"]["patch"]["fits"].get(n) for n in names} if "patch" in s.get("protocols", {}) else None},
        "rows": {"n_rows": int(len(y)), "n_matches": int(len(set(groups.tolist()))), "test_rows": int(len(te)),
                 "test_matches": int(len(set(groups[te].tolist())))},
        "split": split, "settings": inputs, "reference_setting": ref_name, "models": models, "encoder_cfg": encoder_cfg,
        "device": str(device), "cuda": cuda_info, "cuda_max_memory_allocated_mb": max(peak) if peak else None,
        "fits": fits, "bootstrap": boot, "deviations": deviations(args), "references": REFERENCES,
        "outputs": {"preds": str(preds_out), "prediction_caches": str(out_dir)},
        "wall_clock_s": round(time.time() - started, 1),
    }
    np.savez_compressed(preds_out, **store)
    SW.write_json(output, summary)
    log(f"wrote {output} in {summary['wall_clock_s']}s (cuda peak {summary['cuda_max_memory_allocated_mb']} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
