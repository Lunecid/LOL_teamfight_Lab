"""mlp_ablation.py — MLP-on-Same-Features Ablation Study

Representation-vs-Architecture ablation experiment:
  Feed the same engineered time-statistics features (used by LightGBM)
  into a 2-layer MLP to test whether representation design matters
  independently of the model architecture.

If MLP ≈ LightGBM → "representation is the key, architecture is secondary"
If MLP << LightGBM → "tree-based splitting provides additional benefit"

Either outcome strengthens the paper's argument.

Usage:
    python -m analysis.mlp_ablation --out_dir outputs/mlp_ablation [--max_samples N]

Requires a completed LightGBM baseline run with cached data.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MLPAblationResult:
    """Result container for MLP ablation experiment."""
    ok: bool = False
    metrics_train: Dict[str, Any] = None
    metrics_val: Dict[str, Any] = None
    metrics_test: Dict[str, Any] = None
    n_features: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    mlp_config: Dict[str, Any] = None
    training_time_sec: float = 0.0
    epoch_log: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metrics_train is None:
            self.metrics_train = {}
        if self.metrics_val is None:
            self.metrics_val = {}
        if self.metrics_test is None:
            self.metrics_test = {}
        if self.mlp_config is None:
            self.mlp_config = {}
        if self.epoch_log is None:
            self.epoch_log = []


def _build_mlp_model(
    input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.3,
):
    """Build a 2-layer MLP for binary classification.

    Architecture:
        Input(D) -> Linear(D, H) -> ReLU -> Dropout -> Linear(H, H) -> ReLU -> Dropout -> Linear(H, 1)

    This is intentionally simple: the point is to test whether the
    *features* (not the model complexity) drive performance.
    """
    import torch
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    return model


def _standardize(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Xte: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization fit on train, applied to all splits."""
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma < 1e-8] = 1.0  # avoid division by zero for constant features

    Xtr_s = (Xtr - mu) / sigma
    Xva_s = (Xva - mu) / sigma if Xva.size else Xva
    Xte_s = (Xte - mu) / sigma if Xte.size else Xte

    # Replace any remaining NaN/inf from the data
    Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=0.0, neginf=0.0)
    Xva_s = np.nan_to_num(Xva_s, nan=0.0, posinf=0.0, neginf=0.0) if Xva_s.size else Xva_s
    Xte_s = np.nan_to_num(Xte_s, nan=0.0, posinf=0.0, neginf=0.0) if Xte_s.size else Xte_s

    return Xtr_s, Xva_s, Xte_s


def _train_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 15,
    seed: int = 42,
    log_fp: Optional[Path] = None,
):
    """Train a 2-layer MLP with early stopping on validation AUC.

    Returns (model, epoch_log) where epoch_log tracks per-epoch metrics.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    from core.utils import write_log

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = Xtr.shape[1]
    model = _build_mlp_model(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Convert to tensors
    X_train_t = torch.tensor(Xtr, dtype=torch.float32)
    y_train_t = torch.tensor(ytr, dtype=torch.float32)
    X_val_t = torch.tensor(Xva, dtype=torch.float32).to(device) if Xva.size else None
    y_val_np = yva

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0
    epoch_log = []

    n_train = len(X_train_t)

    for epoch in range(max_epochs):
        model.train()
        # Shuffle
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_auc = -1.0
        if X_val_t is not None and len(y_val_np) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).squeeze(-1)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            try:
                val_auc = roc_auc_score(y_val_np, val_probs)
            except ValueError:
                val_auc = 0.5

            scheduler.step(val_auc)

        entry = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 5),
            "val_auc": round(val_auc, 5),
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_log.append(entry)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = (
                f"[MLP] epoch={epoch+1}/{max_epochs} "
                f"loss={avg_loss:.4f} val_auc={val_auc:.4f} "
                f"best={best_val_auc:.4f} patience={patience_counter}/{patience}"
            )
            if log_fp:
                write_log(msg, log_fp)
            else:
                logger.info(msg)

        if patience_counter >= patience:
            msg = f"[MLP] Early stopping at epoch {epoch+1} (best_val_auc={best_val_auc:.4f})"
            if log_fp:
                write_log(msg, log_fp)
            else:
                logger.info(msg)
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model, epoch_log


def _predict_mlp(model, X: np.ndarray) -> np.ndarray:
    """Get predicted probabilities from MLP."""
    import torch

    if X.size == 0:
        return np.array([])

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_t).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def run_mlp_ablation(
    feature_set: str,
    tr_refs: list,
    va_refs: list,
    te_refs: list,
    seed: int,
    log_fp: Path,
    out_dir: Path,
    *,
    max_samples: Optional[int] = None,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 15,
) -> Dict[str, Any]:
    """Run MLP-on-same-features ablation.

    Uses the exact same tabular feature pipeline as LightGBM baseline
    (build_tabular_Xy with seq_to_tabular), then trains a 2-layer MLP.

    Parameters
    ----------
    feature_set : str
        Feature set name (same as baseline).
    tr_refs, va_refs, te_refs : list of FightRef
        Train/val/test split references.
    seed : int
        Random seed.
    log_fp : Path
        Log file path.
    out_dir : Path
        Output directory.
    max_samples : int, optional
        Max samples per split (for quick testing).
    hidden_dim : int
        Hidden layer dimension for MLP.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.

    Returns
    -------
    dict with keys: ok, metrics, mlp_config, training_time_sec, etc.
    """
    from core.utils import write_log, save_json
    from core.config import cfg
    from train.baseline import (
        build_tabular_Xy,
        infer_tabular_plan,
    )
    from core.utils import metrics_from_probs, confusion_from_probs, pretty_cm
    from data.file_io import ensure_dir

    ensure_dir(out_dir)

    out: Dict[str, Any] = {"ok": False}
    write_log("[MLP-ABLATION] Starting MLP-on-same-features ablation", log_fp)

    # ── 1. Build tabular features (identical to LightGBM pipeline) ──
    tab_plan = infer_tabular_plan(tr_refs, feature_set, log_fp=log_fp)
    if tab_plan is None:
        write_log("[MLP-ABLATION] Failed to infer tabular plan", log_fp)
        return out

    write_log(
        f"[MLP-ABLATION] Building tabular features "
        f"(feature_set={feature_set}, seq_key={tab_plan.seq_key})",
        log_fp,
    )

    Xtr, ytr, feat_names, tr_used = build_tabular_Xy(
        tr_refs, feature_set, max_samples=max_samples, log_fp=log_fp, plan=tab_plan,
    )
    Xva, yva, _, va_used = build_tabular_Xy(
        va_refs, feature_set, max_samples=max_samples, log_fp=log_fp, plan=tab_plan,
    )
    Xte, yte, _, te_used = build_tabular_Xy(
        te_refs, feature_set, max_samples=max_samples, log_fp=log_fp, plan=tab_plan,
    )

    if len(tr_used) < 200:
        write_log(f"[MLP-ABLATION] Not enough training samples: {len(tr_used)}", log_fp)
        return out

    n_features_raw = Xtr.shape[1]
    write_log(
        f"[MLP-ABLATION] Data: train={Xtr.shape} val={Xva.shape} test={Xte.shape} "
        f"n_features_raw={n_features_raw}",
        log_fp,
    )

    # ── 2. Apply same constant/quasi-constant feature pruning as LightGBM ──
    _drop_const = bool(getattr(cfg, "DROP_CONSTANT_FEATURES", True))
    _drop_quasi = bool(getattr(cfg, "DROP_QUASI_CONSTANT_FEATURES", True))
    _drop_wfc = bool(getattr(cfg, "DROP_WITHIN_FIGHT_CONSTANT_FEATURES", True))
    if (_drop_const or _drop_quasi or _drop_wfc) and Xtr.shape[1] > 1:
        from core.feature_contract import filter_constant_and_quasi_constant
        const_keep_idx, dc, dq = filter_constant_and_quasi_constant(
            feat_names,
            drop_strictly_constant=_drop_const,
            drop_quasi_constant=_drop_quasi,
            drop_within_fight_constant=_drop_wfc,
        )
        n_dropped = len(dc) + len(dq)
        if n_dropped > 0:
            const_keep = list(const_keep_idx)
            Xtr = Xtr[:, const_keep]
            if Xva.size:
                Xva = Xva[:, const_keep]
            if Xte.size:
                Xte = Xte[:, const_keep]
            feat_names = [feat_names[i] for i in const_keep]
            write_log(
                f"[MLP-ABLATION] const-prune: dropped={n_dropped} kept={len(const_keep)}",
                log_fp,
            )

    # ── 2b. Apply same correlation pruning as LightGBM ──
    if bool(getattr(cfg, "DROP_CORR_FEATURES", False)) and Xtr.shape[1] > 1:
        from train.baseline import corr_prune_tabular
        from core.utils import sanitize_feature_names
        corr_keep_idx, corr_dropped = corr_prune_tabular(
            Xtr, feat_names, seed=seed,
            threshold=float(getattr(cfg, "CORR_THRESHOLD", 0.98)),
        )
        Xtr = Xtr[:, corr_keep_idx]
        if Xva.size:
            Xva = Xva[:, corr_keep_idx]
        if Xte.size:
            Xte = Xte[:, corr_keep_idx]
        feat_names = sanitize_feature_names([feat_names[i] for i in corr_keep_idx])
        write_log(
            f"[MLP-ABLATION] corr-prune: kept={len(corr_keep_idx)} dropped={len(corr_dropped)}",
            log_fp,
        )

    # ── 3. Standardize features (critical for MLP, not needed for trees) ──
    Xtr_s, Xva_s, Xte_s = _standardize(Xtr, Xva, Xte)

    write_log(
        f"[MLP-ABLATION] After preprocessing: D={Xtr_s.shape[1]} features",
        log_fp,
    )

    # ── 4. Train 2-layer MLP ──
    mlp_config = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "input_dim": Xtr_s.shape[1],
        "architecture": "Linear(D,H)->ReLU->Drop->Linear(H,H)->ReLU->Drop->Linear(H,1)",
    }
    write_log(f"[MLP-ABLATION] MLP config: {mlp_config}", log_fp)

    t0 = time.time()
    model, epoch_log = _train_mlp(
        Xtr_s, ytr, Xva_s, yva,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        log_fp=log_fp,
    )
    training_time = time.time() - t0

    write_log(f"[MLP-ABLATION] Training completed in {training_time:.1f}s", log_fp)

    # ── 5. Evaluate ──
    thr = float(getattr(cfg, "CLS_THRESHOLD", 0.5))

    p_tr = _predict_mlp(model, Xtr_s)
    p_va = _predict_mlp(model, Xva_s)
    p_te = _predict_mlp(model, Xte_s)

    met_tr = metrics_from_probs(ytr, p_tr, threshold=thr)
    met_va = metrics_from_probs(yva, p_va, threshold=thr) if Xva.size else {}
    met_te = metrics_from_probs(yte, p_te, threshold=thr) if Xte.size else {}

    write_log(
        f"[MLP-ABLATION] Train: auc={met_tr.get('auc', 0):.4f} "
        f"{pretty_cm(confusion_from_probs(ytr, p_tr, thr))}",
        log_fp,
    )
    if Xva.size:
        write_log(
            f"[MLP-ABLATION] Val  : auc={met_va.get('auc', 0):.4f} "
            f"{pretty_cm(confusion_from_probs(yva, p_va, thr))}",
            log_fp,
        )
    if Xte.size:
        write_log(
            f"[MLP-ABLATION] Test : auc={met_te.get('auc', 0):.4f} "
            f"{pretty_cm(confusion_from_probs(yte, p_te, thr))}",
            log_fp,
        )

    # ── 6. Bootstrap CI ──
    bootstrap_ci_results: Dict[str, Any] = {}
    try:
        from app.experiment_stats import bootstrap_auc_ci as _boot_auc_ci

        if Xva.size and len(yva) >= 20:
            _auc, _ci_lo, _ci_hi = _boot_auc_ci(yva, p_va, n_bootstrap=2000, alpha=0.05, seed=seed)
            bootstrap_ci_results["val"] = {
                "auc": float(_auc),
                "ci_low": float(_ci_lo),
                "ci_high": float(_ci_hi),
                "ci_width": float(_ci_hi - _ci_lo),
            }
            write_log(
                f"[MLP-ABLATION][BOOTSTRAP] val AUC={_auc:.4f} "
                f"95% CI=[{_ci_lo:.4f}, {_ci_hi:.4f}]",
                log_fp,
            )
        if Xte.size and len(yte) >= 20:
            _auc, _ci_lo, _ci_hi = _boot_auc_ci(yte, p_te, n_bootstrap=2000, alpha=0.05, seed=seed)
            bootstrap_ci_results["test"] = {
                "auc": float(_auc),
                "ci_low": float(_ci_lo),
                "ci_high": float(_ci_hi),
                "ci_width": float(_ci_hi - _ci_lo),
            }
            write_log(
                f"[MLP-ABLATION][BOOTSTRAP] test AUC={_auc:.4f} "
                f"95% CI=[{_ci_lo:.4f}, {_ci_hi:.4f}]",
                log_fp,
            )
    except Exception as e:
        write_log(f"[MLP-ABLATION][BOOTSTRAP] CI failed: {e}", log_fp)

    # ── 7. Save report ──
    report = {
        "ok": True,
        "experiment": "mlp_on_same_features",
        "purpose": (
            "Test whether representation design matters independently of "
            "architecture by feeding the same engineered time-statistics "
            "features into a 2-layer MLP instead of LightGBM."
        ),
        "interpretation": {
            "mlp_approx_lgbm": (
                "If MLP achieves comparable AUC to LightGBM, the engineered "
                "features (representation) are the primary driver of performance, "
                "not the tree-based architecture."
            ),
            "mlp_below_lgbm": (
                "If MLP underperforms LightGBM, tree-based splitting provides "
                "additional benefit beyond the feature representation — but the "
                "representation still lifts MLP above raw-input deep models."
            ),
        },
        "mlp_config": mlp_config,
        "metrics": {
            "train": met_tr,
            "val": met_va,
            "test": met_te,
        },
        "bootstrap_ci": bootstrap_ci_results if bootstrap_ci_results else None,
        "n_features_raw": n_features_raw,
        "n_features_after_pruning": Xtr_s.shape[1],
        "feature_pipeline_note": (
            "Identical to LightGBM: build_tabular_Xy -> "
            "constant/quasi-constant pruning -> correlation pruning. "
            "Only additional step is z-score standardization for MLP."
        ),
        "n_samples": {
            "train": len(tr_used),
            "val": len(va_used),
            "test": len(te_used),
        },
        "training_time_sec": round(training_time, 1),
        "n_epochs_trained": len(epoch_log),
        "best_val_auc_epoch": (
            max(epoch_log, key=lambda e: e["val_auc"])["epoch"]
            if epoch_log else None
        ),
        "epoch_log": epoch_log,
        "seed": seed,
    }

    save_json(out_dir / "mlp_ablation_report.json", report)
    write_log(
        f"[MLP-ABLATION] Report saved to {out_dir / 'mlp_ablation_report.json'}",
        log_fp,
    )

    out.update(report)
    return out


def run_mlp_ablation_multi_seed(
    feature_set: str,
    tr_refs: list,
    va_refs: list,
    te_refs: list,
    seeds: List[int],
    log_fp: Path,
    out_dir: Path,
    *,
    max_samples: Optional[int] = None,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 15,
) -> Dict[str, Any]:
    """Run MLP ablation across multiple seeds and aggregate results.

    Returns a summary with per-seed results plus aggregated statistics.
    """
    from core.utils import write_log, save_json
    from data.file_io import ensure_dir

    ensure_dir(out_dir)

    write_log(
        f"[MLP-ABLATION] Multi-seed run: seeds={seeds}",
        log_fp,
    )

    per_seed_results = {}
    val_aucs = []
    test_aucs = []

    for s in seeds:
        seed_dir = out_dir / f"seed_{s}"
        write_log(f"\n[MLP-ABLATION] === Seed {s} ===", log_fp)

        result = run_mlp_ablation(
            feature_set=feature_set,
            tr_refs=tr_refs,
            va_refs=va_refs,
            te_refs=te_refs,
            seed=s,
            log_fp=log_fp,
            out_dir=seed_dir,
            max_samples=max_samples,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )

        per_seed_results[str(s)] = result

        if result.get("ok", False):
            va_auc = result.get("metrics", {}).get("val", {}).get("auc")
            te_auc = result.get("metrics", {}).get("test", {}).get("auc")
            if va_auc is not None:
                val_aucs.append(float(va_auc))
            if te_auc is not None:
                test_aucs.append(float(te_auc))

    # Aggregate
    summary = {
        "ok": len(val_aucs) > 0,
        "experiment": "mlp_on_same_features_multi_seed",
        "seeds": seeds,
        "n_seeds_completed": len(val_aucs),
        "val_auc_mean": float(np.mean(val_aucs)) if val_aucs else None,
        "val_auc_std": float(np.std(val_aucs)) if len(val_aucs) > 1 else None,
        "val_aucs": val_aucs,
        "test_auc_mean": float(np.mean(test_aucs)) if test_aucs else None,
        "test_auc_std": float(np.std(test_aucs)) if len(test_aucs) > 1 else None,
        "test_aucs": test_aucs,
        "per_seed": per_seed_results,
    }

    save_json(out_dir / "mlp_ablation_summary.json", summary)
    write_log(
        f"[MLP-ABLATION] Multi-seed summary: "
        f"val_auc={summary['val_auc_mean']:.4f}±{summary.get('val_auc_std', 0):.4f} "
        f"test_auc={summary.get('test_auc_mean', 0):.4f}±{summary.get('test_auc_std', 0):.4f}",
        log_fp,
    )

    return summary


# ── CLI entry point ──────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MLP-on-Same-Features Ablation Study",
    )
    parser.add_argument("--out_dir", type=str, default="outputs/mlp_ablation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed run (e.g. '42,123,456')")
    parser.add_argument("--feature_set", type=str, default="v2")
    parser.add_argument("--split_mode", type=str, default="patch")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = out_dir / "mlp_ablation.log"

    # Build refs
    try:
        from data.index_split import build_split_refs
        from data.cache_io import load_dataset_index
        idx = load_dataset_index()
        if not idx:
            print("[ERROR] No dataset index. Run cache builder first.")
            return
        tr_refs, va_refs, te_refs = build_split_refs(
            idx, mode=args.split_mode, seed=args.seed,
        )
    except Exception as e:
        print(f"[ERROR] Cannot build refs: {e}")
        return

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        result = run_mlp_ablation_multi_seed(
            feature_set=args.feature_set,
            tr_refs=tr_refs,
            va_refs=va_refs,
            te_refs=te_refs,
            seeds=seeds,
            log_fp=log_fp,
            out_dir=out_dir,
            max_samples=args.max_samples,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
    else:
        result = run_mlp_ablation(
            feature_set=args.feature_set,
            tr_refs=tr_refs,
            va_refs=va_refs,
            te_refs=te_refs,
            seed=args.seed,
            log_fp=log_fp,
            out_dir=out_dir,
            max_samples=args.max_samples,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

    if result.get("ok"):
        print("\n" + "=" * 60)
        print("MLP-on-Same-Features Ablation — Results")
        print("=" * 60)
        metrics = result.get("metrics", {})
        if "val" in metrics and metrics["val"]:
            print(f"  Val AUC:  {metrics['val'].get('auc', 'N/A')}")
        if "test" in metrics and metrics["test"]:
            print(f"  Test AUC: {metrics['test'].get('auc', 'N/A')}")
        if "val_auc_mean" in result:
            print(f"  Val AUC (mean±std): {result['val_auc_mean']:.4f}±{result.get('val_auc_std', 0):.4f}")
            print(f"  Test AUC (mean±std): {result.get('test_auc_mean', 0):.4f}±{result.get('test_auc_std', 0):.4f}")
        print("=" * 60)
    else:
        print("[FAILED] MLP ablation did not complete successfully.")


if __name__ == "__main__":
    main()
