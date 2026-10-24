"""q fitting: STOP-selected epochs/rounds, fresh REFIT, scoring and reload (ARCHITECTURE_KO.md section 4.4).

Rules (EXECUTION_CARDS.json ``optimizer``; FT-Transformer exceptions in ``q.auxiliary_families``):
  - neural: AdamW lr 1e-3, betas (0.9, 0.999), eps 1e-8, constant schedule, gradient-norm clip 1,
    batch 512, at most 100 epochs, patience 12, improvement 1e-5; the batch loss is
    mean(normalized_weight * BCE) (never divided by the batch weight sum); the STOP metric is the
    raw match Brier; best epoch = earliest epoch improving by more than 1e-5;
  - REFIT: a newly seeded model and optimizer on all TRAIN (REFIT_TRAIN preprocessing) for exactly
    the best epoch count; trees refit with the best round count;
  - LightGBM: lr 0.05, feature_fraction 0.8, bagging 1/0, max_bin 255, at most 2000 rounds,
    patience 100, early stopping on the STOP raw match Brier (custom metric, D-Q10);
  - logistic: fit once on all TRAIN with normalized weights; non-convergence is a recorded terminal
    failure that is never selectable (EXECUTION_CARDS.json ``optimizer.linear``; DECISIONS D-Q1).
Exact resume: after every epoch the trainer commits the phase state (epochs, STOP metrics, the last
checkpoint and its sha256, every RNG state); a restart continues from the last verified epoch.
Q_DEV and Q_CAL are scored with the REFIT model (as V scores V_DEV, studies/v_production_20261004/
train.py line 1779); TEST is never scored here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

VERSION = "q_train_v1_20261005"
SEED = 20261001


@dataclass(frozen=True)
class NeuralRules:
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 12
    min_delta: float = 1e-5
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    time_limit_s: float = 24 * 3600.0     # per fit phase, summed over resumed epochs (DECISIONS D-Q4)


class FitTimeLimit(RuntimeError):
    """A FIT or REFIT phase used more than its time limit: a recorded terminal failure (D-Q4)."""


def neural_rules(family: str, weight_decay: float | None = None) -> NeuralRules:
    if family == "ft_transformer":
        return NeuralRules(batch_size=16, max_epochs=60, patience=8, lr=1e-4, weight_decay=1e-5)
    return NeuralRules(weight_decay=float(weight_decay or 0.0))


@dataclass
class Batch:
    inputs: Any
    y: np.ndarray            # 0/1
    w_norm: np.ndarray       # training weight n_rows/(n_matches*m) of the scope
    w_raw: np.ndarray        # 1/m, for match-weighted metrics
    rows: np.ndarray         # q row ids


Source = Callable[[], Iterable[Batch]]   # a fresh, deterministic pass over one split


# ----------------------------------------------------------------------------- io helpers
def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace(pending: Path, target: Path) -> None:
    for attempt in range(41):
        try:
            os.replace(pending, target)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 40:
                raise
            time.sleep(0.05)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(pending, path)


def write_torch(path: Path, payload: Mapping[str, Any]) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(pending, path)
    return sha256_file(path)


def configure_runtime(seed: int = SEED) -> None:
    """Deterministic torch (same flags as the V runner); CUBLAS workspace must be set before CUDA starts."""

    import torch

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def _rng_payload() -> dict[str, Any]:
    import torch

    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(), "numpy": np.random.get_state()}


def _restore_rng(payload: Mapping[str, Any]) -> None:
    import torch

    torch.set_rng_state(payload["torch"].cpu())
    if payload.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda"]])
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])


# ----------------------------------------------------------------------------- metrics
def match_brier(p: np.ndarray, y: np.ndarray, w_raw: np.ndarray) -> float:
    """Raw match-weighted Brier: sum w (p - y)^2 / sum w with w = 1/m (mean over matches of the mean case loss)."""

    p, y, w = (np.asarray(v, dtype=np.float64) for v in (p, y, w_raw))
    return float((w * (p - y) ** 2).sum() / w.sum())


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ----------------------------------------------------------------------------- neural
def _epoch(model, optimizer, source: Source, forward, device, rules: NeuralRules) -> float:
    import torch

    model.train()
    total, weight = 0.0, 0.0
    for batch in source():
        logits = forward(model, batch.inputs, device)
        y = torch.as_tensor(batch.y, dtype=torch.float32, device=device)
        w = torch.as_tensor(batch.w_norm, dtype=torch.float32, device=device)
        loss = torch.mean(w * torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), rules.grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(batch.y)
        weight += len(batch.y)
    return total / max(weight, 1.0)


def predict(model, source: Source, forward, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(rows, logits float64, y, w_raw) in source order."""

    import torch

    model.eval()
    rows, logits, ys, ws = [], [], [], []
    with torch.no_grad():
        for batch in source():
            logits.append(forward(model, batch.inputs, device).double().cpu().numpy())
            rows.append(batch.rows); ys.append(batch.y); ws.append(batch.w_raw)
    return (np.concatenate(rows), np.concatenate(logits), np.concatenate(ys), np.concatenate(ws))


def _optimizer(model, rules: NeuralRules):
    import torch

    return torch.optim.AdamW(model.parameters(), lr=rules.lr, betas=rules.betas, eps=rules.eps, weight_decay=rules.weight_decay)


def _new_model(build: Callable[[], Any], device):
    import torch

    torch.manual_seed(SEED)                  # every phase starts from the same seeded initialization
    return build().to(device)


def _run_phase(phase: str, build, forward, train_source: Source, rules: NeuralRules, out_dir: Path, identity: Mapping[str, Any],
               device, *, epochs: int | None = None, stop_source: Source | None = None) -> dict[str, Any]:
    """FIT (with STOP early stopping) or REFIT (fixed ``epochs``); resumable at epoch granularity."""

    import torch

    state_path = out_dir / f"{phase}_STATE.json"
    checkpoint = out_dir / f"{phase.lower()}_last.pt"
    model = _new_model(build, device)
    optimizer = _optimizer(model, rules)
    state = {"version": VERSION, "phase": phase, "identity": dict(identity), "epochs": [], "done": False}
    if state_path.is_file():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved.get("identity") != dict(identity):
            raise ValueError(f"{phase} state belongs to another job; archive {out_dir} before rerunning")
        if saved.get("done"):
            return saved
        if saved["epochs"]:
            last = saved["epochs"][-1]
            if not checkpoint.is_file() or sha256_file(checkpoint) != last["checkpoint_sha256"]:
                raise ValueError(f"{phase} checkpoint does not verify; archive {out_dir} before rerunning")
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            _restore_rng(payload["rng"])
            state = saved
    best_epoch = state.get("best_epoch", 0)
    best = state.get("best_stop_brier", float("inf"))
    limit = rules.max_epochs if epochs is None else int(epochs)
    for epoch in range(len(state["epochs"]) + 1, limit + 1):
        started = time.time()
        train_loss = _epoch(model, optimizer, train_source, forward, device, rules)
        record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - started}
        if stop_source is not None:
            _, logits, y, w = predict(model, stop_source, forward, device)
            record["stop_brier"] = match_brier(sigmoid(logits), y, w)
            if record["stop_brier"] < best - rules.min_delta:
                best, best_epoch = record["stop_brier"], epoch
        record["checkpoint_sha256"] = write_torch(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                                               "rng": _rng_payload(), "epoch": epoch})
        state["epochs"].append(record)
        state.update(best_epoch=best_epoch, best_stop_brier=best if stop_source is not None else None)
        stop = stop_source is not None and epoch - best_epoch >= rules.patience
        state["done"] = phase == "FIT" and (stop or epoch == limit)
        used = sum(float(record["seconds"]) for record in state["epochs"])
        if not state["done"] and epoch < limit and used > rules.time_limit_s:
            state["time_limit"] = {"used_s": used, "limit_s": rules.time_limit_s, "epoch": epoch}
            atomic_json(state_path, state)
            raise FitTimeLimit(f"{phase} used {used:.0f} s over {epoch} epochs, above the {rules.time_limit_s:.0f} s limit")
        atomic_json(state_path, state)
        if stop:
            break
    if phase == "REFIT":
        # Done only once the final model is durable, so a crash in between redoes just this write.
        if len(state["epochs"]) != limit:
            raise RuntimeError("REFIT ended before its fixed epoch count")
        state["model_sha256"] = write_torch(out_dir / "model_refit.pt", {"model": model.state_dict()})
    state["done"] = True
    atomic_json(state_path, state)
    return state


def fit_neural(build: Callable[[], Any] | Mapping[str, Callable[[], Any]], forward, sources: Mapping[str, Source], rules: NeuralRules,
               out_dir: Path, identity: Mapping[str, Any], device: str = "cpu") -> dict[str, Any]:
    """FIT (fit rows, STOP early stopping) -> REFIT (all TRAIN, best epochs) -> score Q_DEV/Q_CAL -> reload check.

    ``sources``: fit, stop (FIT-scope inputs), refit, q_dev, q_cal (REFIT-scope inputs).  ``build`` is one
    builder or {"FIT": ..., "REFIT": ...}: the two scopes have their own vocabularies (and table widths),
    so the REFIT model is built for the REFIT preprocessing."""

    import torch

    builds = dict(build) if isinstance(build, Mapping) else {"FIT": build, "REFIT": build}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    fit = _run_phase("FIT", builds["FIT"], forward, sources["fit"], rules, out_dir, identity, dev, stop_source=sources["stop"])
    if not fit.get("best_epoch"):
        raise RuntimeError("no STOP improvement: best epoch undefined")
    refit = _run_phase("REFIT", builds["REFIT"], forward, sources["refit"], rules, out_dir, identity, dev, epochs=int(fit["best_epoch"]))
    model = _new_model(builds["REFIT"], dev)
    model.load_state_dict(torch.load(out_dir / "model_refit.pt", map_location=dev, weights_only=False)["model"])
    scores = score_splits(model, forward, {name: sources[name] for name in ("q_dev", "q_cal")}, out_dir, dev)
    reload_model = _new_model(builds["REFIT"], dev)
    reload_model.load_state_dict(torch.load(out_dir / "model_refit.pt", map_location=dev, weights_only=False)["model"])
    _, again, _, _ = predict(reload_model, sources["q_dev"], forward, dev)
    stored = np.load(out_dir / "pred_q_dev.npz")["logit"]
    reload = {"max_abs_logit_diff": float(np.abs(again - stored).max()) if len(stored) else 0.0}
    reload["passed"] = reload["max_abs_logit_diff"] <= 1e-7
    if not reload["passed"]:
        raise RuntimeError(f"reload check failed: {reload}")
    return {"status": "completed", "best_epoch": fit["best_epoch"], "best_stop_brier": fit["best_stop_brier"],
            "fit_epochs": len(fit["epochs"]), "refit_epochs": len(refit["epochs"]), "model_sha256": refit["model_sha256"],
            "scores": scores, "reload": reload}


def score_splits(model, forward, sources: Mapping[str, Source], out_dir: Path, device) -> dict[str, Any]:
    from studies.label_production_20261005.assemble import write_npz

    result = {}
    for name, source in sources.items():
        rows, logits, y, w = predict(model, source, forward, device)
        sha = write_npz(Path(out_dir) / f"pred_{name}.npz", {"row": rows, "logit": logits, "p": sigmoid(logits)})
        result[name] = {"rows": int(len(rows)), "raw_match_brier": match_brier(sigmoid(logits), y, w), "pred_sha256": sha}
    return result


# ----------------------------------------------------------------------------- tables
def _stack(source: Source) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy import sparse

    xs, ys, wn, wr, rows = [], [], [], [], []
    for batch in source():
        xs.append(batch.inputs); ys.append(batch.y); wn.append(batch.w_norm); wr.append(batch.w_raw); rows.append(batch.rows)
    return sparse.vstack(xs, format="csr"), np.concatenate(ys), np.concatenate(wn), np.concatenate(wr), np.concatenate(rows)


def brier_eval(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> tuple[str, float, bool]:
    """LightGBM custom metric: raw match-weighted Brier on the STOP set (lower is better)."""

    return "match_brier", match_brier(y_pred, y_true, weight), False


def fit_lightgbm(card: Mapping[str, Any], sources: Mapping[str, Source], out_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    import pickle

    from lightgbm import early_stopping
    from studies.label_production_20261005.assemble import write_npz
    from studies.v_model_fit_input_20261003.models import LightGBMAdapter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "FIT_DONE.json"
    if done.is_file() and json.loads(done.read_text(encoding="utf-8")).get("identity") == dict(identity):
        best = int(json.loads(done.read_text(encoding="utf-8"))["best_iteration"])
    else:
        x, y, wn, _, _ = _stack(sources["fit"])
        xs, ys, _, wrs, _ = _stack(sources["stop"])
        adapter = LightGBMAdapter(card=card, random_state=SEED, num_threads=3)
        adapter.estimator.set_params(metric="None")
        adapter.estimator.fit(x, y, sample_weight=wn, eval_set=[(xs, ys)], eval_sample_weight=[wrs], eval_metric=brier_eval,
                              callbacks=[early_stopping(100, first_metric_only=True, verbose=False)])
        best = int(adapter.estimator.best_iteration_ or adapter.estimator.n_estimators)
        atomic_json(done, {"identity": dict(identity), "best_iteration": best,
                           "best_stop_brier": float(adapter.estimator.best_score_["valid_0"]["match_brier"]), "utc": utc()})
    x, y, wn, _, _ = _stack(sources["refit"])
    adapter = LightGBMAdapter(card=card, random_state=SEED, n_estimators=best, num_threads=3)
    adapter.fit(x, y, sample_weight=wn)
    model_path = out_dir / "model_refit.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(adapter, stream, protocol=pickle.HIGHEST_PROTOCOL)
    scores = {}
    for name in ("q_dev", "q_cal"):
        xq, yq, _, wq, rows = _stack(sources[name])
        logits = adapter.predict_logits(xq)
        sha = write_npz(out_dir / f"pred_{name}.npz", {"row": rows, "logit": logits, "p": sigmoid(logits)})
        scores[name] = {"rows": int(len(rows)), "raw_match_brier": match_brier(sigmoid(logits), yq, wq), "pred_sha256": sha}
    with model_path.open("rb") as stream:
        reloaded = pickle.load(stream)
    xq, _, _, _, _ = _stack(sources["q_dev"])
    diff = float(np.abs(reloaded.predict_logits(xq) - np.load(out_dir / "pred_q_dev.npz")["logit"]).max())
    if diff > 1e-7:
        raise RuntimeError(f"LightGBM reload differs by {diff}")
    return {"status": "completed", "best_iteration": best, "model_sha256": sha256_file(model_path), "scores": scores, "reload": {"max_abs_logit_diff": diff}}


def fit_logistic(card: Mapping[str, Any], sources: Mapping[str, Source], out_dir: Path) -> dict[str, Any]:
    """Logistic-type q model: one fit on all TRAIN; non-convergence is a recorded terminal failure (D-Q1)."""

    import pickle

    from studies.label_production_20261005.assemble import write_npz
    from studies.v_model_fit_input_20261003.models import LogisticAdapter, LogisticNonconvergenceError

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, wn, _, _ = _stack(sources["refit"])
    adapter = LogisticAdapter(C=float(card["C"]), card=card)
    try:
        adapter.fit(x, y, sample_weight=wn)
    except LogisticNonconvergenceError as error:
        return {"status": "failed", "reason": "nonconverged", "terminal": True, "selectable": False,
                "message": str(error), "convergence": getattr(adapter, "convergence_", None)}
    model_path = out_dir / "model_refit.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(adapter, stream, protocol=pickle.HIGHEST_PROTOCOL)
    scores = {}
    for name in ("q_dev", "q_cal"):
        xq, yq, _, wq, rows = _stack(sources[name])
        logits = adapter.predict_logits(xq)
        sha = write_npz(out_dir / f"pred_{name}.npz", {"row": rows, "logit": logits, "p": sigmoid(logits)})
        scores[name] = {"rows": int(len(rows)), "raw_match_brier": match_brier(sigmoid(logits), yq, wq), "pred_sha256": sha}
    return {"status": "completed", "convergence": getattr(adapter, "convergence_", None), "model_sha256": sha256_file(model_path), "scores": scores}
