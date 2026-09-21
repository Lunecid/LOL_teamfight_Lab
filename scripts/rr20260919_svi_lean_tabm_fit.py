#!/usr/bin/env python3
"""Lean Tier-A TabM fit on 352 ridge inputs (protocol §2).

Mirrors Track-A MLP protocol for one new family:
  TRAIN fit (stop10 early-stop + full refit) → Q_CAL calibrators → Q_SELECT selection
  → sealed MAIN_TEST T score vs frozen PT / LightGBM / MLP.

Does NOT touch frozen iq/Track-A trees. Writes:
  outputs/svi_lean_tabm_20260919/{selection,models,eval,REPORT.md,results.json}

Usage:
  python scripts/rr20260919_svi_lean_tabm_fit.py --device cuda
  python scripts/rr20260919_svi_lean_tabm_fit.py --device cuda --smoke   # 1/8 matches, pseudo roles
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_CBWR", "AVX2,STRICT")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from train.svi_tabular_meta import TABM_GRID, TabM, param_count  # noqa: E402


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "incremental_q_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


SEEDS = (7, 42, 123)
OPT = dict(
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,
    max_epochs=100,
    patience=10,
    dtype_train="float32",
)
PRED_CHUNK = 8192
PRED_THREADS = 1
ROLE_TAG = "EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"


def _import_pipeline(data_root: Path):
    scripts = data_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import fc20260915_common as C  # noqa: E402
    import cr20260915_common as K  # noqa: E402
    import iq20260915_common as Q  # noqa: E402
    return C, K, Q


def config_names() -> List[str]:
    return [f"tabm_K{c['k']}_W{c['width']}_D{c['dropout']:g}" for c in TABM_GRID]


def config_params(name: str) -> dict:
    return dict(TABM_GRID[config_names().index(name)])


def candidate_names(cals: Sequence[str]) -> List[str]:
    return sorted(f"{cfg}__{cal}" for cfg in config_names() for cal in cals)


def split_candidate(name: str) -> Tuple[str, str]:
    cfg, cal = name.rsplit("__", 1)
    return cfg, cal


def select_rule(metrics: Dict[str, Tuple[float, float]]) -> str:
    return min(metrics.keys(), key=lambda c: (metrics[c][0], metrics[c][1], c))


def torch_setup(device: str, seed: int):
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    return torch


def weighted_bce_mean(logits, y, w):
    import torch.nn.functional as F

    return (w * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).mean()


def stop_brier_gpu(torch, module, Zs, ys, ws) -> float:
    module.eval()
    with torch.no_grad():
        p = torch.sigmoid(module(Zs).squeeze(1)).detach().cpu().numpy().astype(np.float64)
    y = ys.detach().cpu().numpy().astype(np.float64)
    w = ws.detach().cpu().numpy().astype(np.float64)
    return float(np.average((p - y) ** 2, weights=w))


def run_training(k: int, width: int, dropout: float, Z, y, w, seed, device, epochs,
                 stop=None, patience=None, log=None):
    torch = torch_setup(device, seed)
    module = TabM(Z.shape[1], k=k, width=width, dropout=dropout).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    Zg = torch.as_tensor(np.asarray(Z, dtype=np.float32), device=device)
    yg = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    wg = torch.as_tensor(np.asarray(w, dtype=np.float32), device=device)
    if stop is not None:
        Zs = torch.as_tensor(np.asarray(stop[0], dtype=np.float32), device=device)
        ys = torch.as_tensor(np.asarray(stop[1], dtype=np.float32), device=device)
        ws = torch.as_tensor(np.asarray(stop[2], dtype=np.float32), device=device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n, bs = len(y), OPT["batch_size"]
    curve, best, best_epoch, wait, ep = [], None, 0, 0, 0
    for ep in range(1, int(epochs) + 1):
        module.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            loss = weighted_bce_mean(module(Zg[idx]).squeeze(1), yg[idx], wg[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if stop is not None:
            b = stop_brier_gpu(torch, module, Zs, ys, ws)
            curve.append(b)
            if best is None or b < best:
                best, best_epoch, wait = b, ep, 0
            else:
                wait += 1
            if log and (ep % 10 == 0 or ep == 1):
                log(f"    epoch {ep} stop_brier={b:.6f} best_epoch={best_epoch}")
            if wait >= int(patience):
                break
    return module, curve, best_epoch, ep


def state_to_numpy(module) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in module.state_dict().items()}


def predict_cpu64(k: int, width: int, dropout: float, state: dict, Z: np.ndarray) -> np.ndarray:
    import torch

    torch.set_num_threads(PRED_THREADS)
    module = TabM(Z.shape[1], k=k, width=width, dropout=dropout).double()
    module.load_state_dict({kk: torch.from_numpy(np.asarray(vv, dtype=np.float32)).double()
                            for kk, vv in state.items()})
    module.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), PRED_CHUNK):
            zc = torch.from_numpy(np.ascontiguousarray(Z[i : i + PRED_CHUNK], dtype=np.float64))
            out.append(torch.sigmoid(module(zc).squeeze(1)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


class TabMQBase:
    """Three-seed TabM on 352 ridge (imputer+scaler TRAIN-fitted; stop10 then full refit)."""

    def __init__(self, config: str, input_names: Sequence[str], device: str = "cuda",
                 seeds=SEEDS, max_epochs=None, patience=None):
        self.config = config
        self.params = config_params(config)
        self.input_names = list(input_names)
        self.seeds = tuple(seeds)
        self.device = device
        self.max_epochs = int(OPT["max_epochs"] if max_epochs is None else max_epochs)
        self.patience = int(OPT["patience"] if patience is None else patience)

    def _pre(self, X):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        imp = SimpleImputer(strategy="median").fit(X)
        sc = StandardScaler().fit(imp.transform(X))
        return imp, sc

    def fit(self, Xin, y, g, Q, log=None):
        Xin = np.asarray(Xin, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        g = np.asarray(g).astype(str)
        if not np.isfinite(Xin).all():
            raise ValueError("non-finite TRAIN inputs")
        stop = Q.stop_mask(g)
        fit90 = ~stop
        w_fit, w_stop, w_all = Q.weights(g[fit90]), Q.weights(g[stop]), Q.weights(g)
        imp90, sc90 = self._pre(Xin[fit90])
        Z90 = sc90.transform(imp90.transform(Xin[fit90]))
        Zst = sc90.transform(imp90.transform(Xin[stop]))
        self.imputer, self.scaler = self._pre(Xin)
        Zall = self.scaler.transform(self.imputer.transform(Xin))
        k, width, dropout = self.params["k"], self.params["width"], self.params["dropout"]
        self.stop_record = dict(
            fit90=Q.weight_record(g[fit90], w_fit),
            stop10=Q.weight_record(g[stop], w_stop),
            full=Q.weight_record(g, w_all),
            patience=self.patience,
            max_epochs=self.max_epochs,
            device=self.device,
            optimizer=OPT,
            seeds={},
        )
        self.states, self.arch = [], None
        for seed in self.seeds:
            t0 = time.time()
            m_stop, curve, best, _ = run_training(
                k, width, dropout, Z90, y[fit90], w_fit, seed, self.device,
                self.max_epochs, stop=(Zst, y[stop], w_stop), patience=self.patience, log=log)
            if best < 1:
                raise RuntimeError("no epoch recorded in stop phase")
            stop_s = time.time() - t0
            t1 = time.time()
            m_final, _, _, ran_final = run_training(
                k, width, dropout, Zall, y, w_all, seed, self.device, best)
            self.states.append(state_to_numpy(m_final))
            if self.arch is None:
                self.arch = dict(n_params=param_count(m_final), repr=str(m_final))
            self.stop_record["seeds"][str(seed)] = dict(
                best_epoch=int(best),
                epochs_evaluated=len(curve),
                best_stop_weighted_brier=float(curve[best - 1]),
                stop_curve=[float(v) for v in curve],
                refit_epochs=int(ran_final),
                stop_seconds=round(stop_s, 2),
                refit_seconds=round(time.time() - t1, 2),
            )
            if log:
                log(f"{self.config} seed {seed}: best_epoch={best} stop_brier={curve[best - 1]:.6f} "
                    f"stop {stop_s:.1f}s refit {time.time() - t1:.1f}s")
            del m_stop, m_final
        return self

    def design(self, Xin):
        Xin = np.asarray(Xin, dtype=np.float64)
        return self.scaler.transform(self.imputer.transform(Xin))

    def raw(self, Xin):
        Z = self.design(Xin)
        k, width, dropout = self.params["k"], self.params["width"], self.params["dropout"]
        return np.stack([predict_cpu64(k, width, dropout, st, Z) for st in self.states]).mean(axis=0)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_select(data_root: Path, out_dir: Path, device: str, smoke: bool) -> Dict[str, Any]:
    C, K, Q = _import_pipeline(data_root)
    import joblib

    schema = C.read_json(Q.FC / "q_pre_only_schema.json")
    ridge = list(schema["predictor_sets"]["ridge"])
    if len(ridge) != Q.EXPECTED_RIDGE_COUNT:
        raise SystemExit(f"ridge count {len(ridge)} != {Q.EXPECTED_RIDGE_COUNT}")

    _log(f"loading trainval cohort=T smoke={smoke}")
    # load_trainval first arg is unused for paths when smoke=False — base=Q.OUT is conventional
    D = Q.load_trainval(Q.OUT if not smoke else Q.SMOKE, smoke=smoke, cohort="T")
    names = D["names"]
    if names != schema["input_names_all"] or C.q_feature_sets(names)["ridge"] != ridge:
        raise SystemExit("input schema mismatch vs q_pre_only_schema.json")
    X_all, y, g = D["X"], D["y"], D["g"]
    cols = [names.index(n) for n in ridge]
    X = X_all[:, cols]
    M = {k: D["role"] == k for k in ("TRAIN", "Q_CAL", "Q_SELECT")}
    for a, b in (("TRAIN", "Q_CAL"), ("TRAIN", "Q_SELECT"), ("Q_CAL", "Q_SELECT")):
        if set(g[M[a]]) & set(g[M[b]]):
            raise SystemExit(f"match overlap {a}/{b}")
    W = {k: Q.weights(g[M[k]]) for k in M}
    counts = {
        k: dict(
            rows=int(m.sum()),
            matches=int(len(np.unique(g[m]))),
            positive_rate=float(np.average(y[m], weights=W[k])),
        )
        for k, m in M.items()
    }
    _log(f"counts={counts}")

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    sel_path = out_dir / "selection" / "tabm_T.json"
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    if sel_path.exists():
        _log(f"resume: {sel_path} exists — skip fit")
        return json.loads(sel_path.read_text(encoding="utf-8"))

    cals = list(K.CALS)
    cands = candidate_names(cals)
    P: Dict[str, Dict[str, np.ndarray]] = {}
    fitrec, calrec, hashes, stoprec = {}, {}, {}, {}
    Xtr = X[M["TRAIN"]]

    for i, cfg in enumerate(config_names()):
        _log(f"fit {i + 1}/{len(config_names())}: {cfg}")
        basem = TabMQBase(cfg, ridge, device=device).fit(Xtr, y[M["TRAIN"]], g[M["TRAIN"]], Q, log=_log)
        joblib.dump(basem, models_dir / f"{cfg}.joblib")
        raws = {role: basem.raw(X[M[role]]) for role in M}
        cals_fit = K.fit_calibrators(raws["Q_CAL"], y[M["Q_CAL"]], W["Q_CAL"])
        calrec[cfg] = Q.calibrator_record(cals_fit, raws["Q_CAL"])
        stoprec[cfg] = basem.stop_record
        fitrec[cfg] = dict(arch=basem.arch, params=basem.params, n_params=basem.arch["n_params"])
        for cal in cals:
            cand = f"{cfg}__{cal}"
            if cal == "raw":
                P[cand] = {role: raws[role] for role in M}
            else:
                P[cand] = {
                    role: K.apply_calibration(cal, cals_fit[cal], raws[role]) for role in M
                }

    eligible = {
        c: bool(all(np.isfinite(P[c][k]).all() for k in ("Q_CAL", "Q_SELECT")))
        for c in cands
    }
    met = {
        c: {role: C.evaluate(y[M[role]], P[c][role], g[M[role]], bins=(role == "Q_SELECT"))
            for role in M}
        for c in cands
    }
    elig = {c: (met[c]["Q_SELECT"]["brier"], met[c]["Q_SELECT"]["logloss"]) for c in cands if eligible[c]}
    if not elig:
        raise SystemExit("no eligible TabM candidates")
    chosen = select_rule(elig)
    ranking = sorted(elig.keys(), key=lambda c: (elig[c][0], elig[c][1], c))
    chosen_cfg, chosen_cal = split_candidate(chosen)

    sel = dict(
        role=ROLE_TAG,
        family="tabm",
        cohort="T",
        horizon_s=90,
        smoke=smoke,
        rule="lowest Q_SELECT match-weighted Brier, then log loss, then name",
        candidates_lexical_tie_order=cands,
        eligible=eligible,
        chosen=chosen,
        chosen_config=chosen_cfg,
        chosen_calibration=chosen_cal,
        ranking=ranking,
        select_metrics={
            c: {k: met[c]["Q_SELECT"][k] for k in ("brier", "logloss", "auc", "intercept", "slope", "ece_10bin")}
            for c in cands
        },
        calibrate_metrics={c: {k: met[c]["Q_CAL"][k] for k in ("brier", "logloss", "auc")} for c in cands},
        split_counts=counts,
        fit_records=fitrec,
        calibrators=calrec,
        stop_records=stoprec,
        declared_grid=[dict(name=n, **config_params(n)) for n in config_names()],
        optimizer=OPT,
        seeds=list(SEEDS),
        n_inputs=len(ridge),
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    )
    C.write_json(sel_path, sel)
    (out_dir / "internal_stop" / "tabm_T.json").parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out_dir / "internal_stop" / "tabm_T.json", stoprec)
    _log(f"chosen={chosen} Q_SELECT Brier={met[chosen]['Q_SELECT']['brier']:.6f}")
    return sel


def sealed_eval(data_root: Path, out_dir: Path, sel: Dict[str, Any], smoke: bool) -> Dict[str, Any]:
    """Score chosen TabM on MAIN_TEST T; contrast vs iq PT / LightGBM and Track-A MLP."""
    C, K, Q = _import_pipeline(data_root)
    import joblib
    from sklearn.metrics import roc_auc_score

    if smoke:
        return dict(status="skipped", reason="smoke has no sealed MAIN_TEST")

    schema = C.read_json(Q.FC / "q_pre_only_schema.json")
    ridge = list(schema["predictor_sets"]["ridge"])
    cfg = sel["chosen_config"]
    cal = sel["chosen_calibration"]
    basem: TabMQBase = joblib.load(out_dir / "models" / f"{cfg}.joblib")

    F, Lb, Co, _ = Q.load_parent_set("MAIN_TEST", Q.OUT, "lean TabM sealed eval MAIN_TEST")
    names = list(F["input_names"]) if "input_names" in F else list(schema["input_names_all"])
    # load_parent_set returns feature dict with X_input
    m = (Lb["valid_h90"] == 1) & (Co["cohort"] == Q.COHORT_CODE["T"])
    g = F["match"][m].astype(str)
    y = Lb["Y_h90"][m].astype(int)
    X = F["X_input"][m][:, [names.index(n) for n in ridge]]
    w = Q.weights(g)
    raw = basem.raw(X)
    # reload calibrators from Q_CAL by re-fitting is wrong; load from selection calibrator objects
    # Re-fit calibrators on Q_CAL using saved model (same as selection)
    D = Q.load_trainval(Q.OUT, smoke=False, cohort="T")
    cols = [D["names"].index(n) for n in ridge]
    Mcal = D["role"] == "Q_CAL"
    raw_cal = basem.raw(D["X"][Mcal][:, cols])
    cals_fit = K.fit_calibrators(raw_cal, D["y"][Mcal], Q.weights(D["g"][Mcal]))
    if cal == "raw":
        p = raw
    else:
        p = K.apply_calibration(cal, cals_fit[cal], raw)

    brier = float(np.average((p - y) ** 2, weights=w))
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        auc = float("nan")

    # Load sealed iq / Track-A named preds for same rows
    contrasts = {}
    iq_pred = Q.OUT / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    if iq_pred.is_file():
        z = np.load(iq_pred, allow_pickle=False)
        # align by (match, s_ms)
        keys_te = list(zip(g.tolist(), F["s_ms"][m].astype(np.int64).tolist()))
        keys_z = list(zip(z["match"].astype(str).tolist(), z["s_ms"].astype(np.int64).tolist()))
        idx = {k: i for i, k in enumerate(keys_z)}
        order = np.array([idx[k] for k in keys_te])
        for name in ("pt_winner", "lgbm_winner", "logit_winner"):
            key = f"named__{name}"
            if key in z.files:
                pref = z[key][order]
                contrasts[f"tabm_minus_{name}"] = dict(
                    delta_brier=float(np.average((p - y) ** 2, weights=w)
                                      - np.average((pref - y) ** 2, weights=w)),
                    ref_brier=float(np.average((pref - y) ** 2, weights=w)),
                )

    ta_pred = data_root / "outputs" / "track_a_mlp_20260916" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    if ta_pred.is_file():
        z = np.load(ta_pred, allow_pickle=False)
        keys_z = list(zip(z["match"].astype(str).tolist(), z["s_ms"].astype(np.int64).tolist()))
        idx = {k: i for i, k in enumerate(keys_z)}
        keys_te = list(zip(g.tolist(), F["s_ms"][m].astype(np.int64).tolist()))
        order = np.array([idx[k] for k in keys_te])
        for name in ("mlp_winner", "resmlp_winner"):
            key = f"named__{name}"
            if key in z.files:
                pref = z[key][order]
                contrasts[f"tabm_minus_{name}"] = dict(
                    delta_brier=float(np.average((p - y) ** 2, weights=w)
                                      - np.average((pref - y) ** 2, weights=w)),
                    ref_brier=float(np.average((pref - y) ** 2, weights=w)),
                )

    # B40 subset
    p_pre = Lb["p_pre"][m].astype(float)
    b40 = (p_pre >= 0.4) & (p_pre <= 0.6)
    b40_out = None
    if b40.any() and iq_pred.is_file():
        z = np.load(iq_pred, allow_pickle=False)
        keys_te = list(zip(g.tolist(), F["s_ms"][m].astype(np.int64).tolist()))
        keys_z = list(zip(z["match"].astype(str).tolist(), z["s_ms"].astype(np.int64).tolist()))
        idx = {k: i for i, k in enumerate(keys_z)}
        order = np.array([idx[k] for k in keys_te])
        pt = z["named__pt_winner"][order]
        w40 = Q.weights(g[b40])
        b40_out = dict(
            rows=int(b40.sum()),
            tabm_brier=float(np.average((p[b40] - y[b40]) ** 2, weights=w40)),
            pt_brier=float(np.average((pt[b40] - y[b40]) ** 2, weights=w40)),
            delta_brier_tabm_minus_pt=float(
                np.average((p[b40] - y[b40]) ** 2, weights=w40)
                - np.average((pt[b40] - y[b40]) ** 2, weights=w40)
            ),
        )

    pred_path = out_dir / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_path,
        match=g,
        s_ms=F["s_ms"][m].astype(np.int64),
        y=y,
        tabm_winner=p.astype(np.float64),
        p_pre=p_pre,
    )

    out = dict(
        set="MAIN_TEST",
        cohort="T",
        rows=int(len(y)),
        matches=int(len(np.unique(g))),
        chosen=sel["chosen"],
        brier=brier,
        auc=auc,
        contrasts=contrasts,
        B40=b40_out,
        predictions=str(pred_path),
        epistemic=ROLE_TAG,
    )
    C.write_json(out_dir / "eval" / "results.json", out)
    return out


def write_report(out_dir: Path, sel: Dict[str, Any], sealed: Optional[Dict[str, Any]]) -> None:
    lines = [
        "# Lean Tier-A TabM reselection (352 ridge)",
        "",
        f"Generated: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "",
        f"**Epistemic:** {ROLE_TAG}",
        "",
        "## Selection (Q_SELECT T)",
        "",
        f"- Chosen: `{sel['chosen']}`",
        f"- Q_SELECT Brier: {sel['select_metrics'][sel['chosen']]['brier']:.6f}",
        f"- Q_SELECT AUC: {sel['select_metrics'][sel['chosen']]['auc']:.4f}",
        "",
        "| Candidate | Q_SELECT Brier | AUC |",
        "|---|---:|---:|",
    ]
    for c in sel["ranking"][:12]:
        m = sel["select_metrics"][c]
        lines.append(f"| {c} | {m['brier']:.6f} | {m['auc']:.4f} |")
    lines += ["", "## Sealed MAIN_TEST T (15.16)", ""]
    if not sealed or sealed.get("status") == "skipped":
        lines.append(f"_Skipped:_ {(sealed or {}).get('reason', 'n/a')}")
    else:
        lines.append(f"- n={sealed['rows']} / matches={sealed['matches']}")
        lines.append(f"- TabM Brier={sealed['brier']:.6f} AUC={sealed['auc']:.4f}")
        lines.append("")
        lines.append("| Contrast | ΔBrier (TabM − ref) |")
        lines.append("|---|---:|")
        for k, v in sealed.get("contrasts", {}).items():
            lines.append(f"| {k} | {v['delta_brier']:+.6f} |")
        if sealed.get("B40"):
            b = sealed["B40"]
            lines += [
                "",
                f"**B40** (n={b['rows']}): TabM−PT ΔBrier = {b['delta_brier_tabm_minus_pt']:+.6f}",
            ]
    lines += [
        "",
        "## Protocol notes",
        "",
        "- Primary paper contrast remains **q − PT** (LightGBM sealed winner until TabM beats it on Q_SELECT *and* sealed).",
        "- Same 15.16 T rows for all sealed numbers above.",
        "- Limited search budget: 6 TabM configs × 3 seeds × Track-A OPT.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_lean_tabm_20260919")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-sealed", action="store_true")
    args = ap.parse_args(argv)

    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        out_dir = out_dir / "smoke_train_only"
        out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"data_root={data_root}")
    _log(f"out_dir={out_dir} device={args.device}")

    t0 = time.time()
    sel = fit_select(data_root, out_dir, args.device, args.smoke)
    sealed = None
    if not args.skip_sealed:
        _log("sealed MAIN_TEST T eval…")
        sealed = sealed_eval(data_root, out_dir, sel, args.smoke)
    write_report(out_dir, sel, sealed)
    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        seconds=round(time.time() - t0, 1),
        selection=dict(chosen=sel["chosen"], q_select_brier=sel["select_metrics"][sel["chosen"]]["brier"]),
        sealed=sealed,
        protocol="docs/SVI_MODEL_RESELECTION_TRANSFER_20260919.md §2 Tier A",
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"done in {time.time() - t0:.1f}s → {out_dir}")
    print(json.dumps({"chosen": sel["chosen"], "sealed_brier": (sealed or {}).get("brier")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
