"""Run one q job spec (ARCHITECTURE_KO.md sections 4.4, 6.2, 6.3).

A spec (written by reports/q_production_20261005/q_supervisor.py) names one resolved job, the
variant's prepared inputs, the device and the q scientific release.  The worker verifies the
release, loads the saved scope contexts (never refits them), builds the five sources
  fit   FIT context, TRAIN rows outside the q STOP matches, weights n/(n_matches*m) over them
  stop  FIT context, q STOP rows, raw weights 1/m (the STOP metric is the raw match Brier)
  refit REFIT_TRAIN context, all TRAIN rows, weights over all TRAIN
  q_dev, q_cal  REFIT_TRAIN context, raw weights
and dispatches by family:
  neural (G0, G1, tied, GraphSAGE, shared slot, plain MLP, FT, pure set) -> qtrain.fit_neural,
    one builder per scope (vocabularies and table widths differ); G0 and tied widths are matched
    to G1 at the same H within 5 % total parameters on the REFIT_TRAIN encoder;
  lightgbm -> qtrain.fit_lightgbm;  logistic -> qtrain.fit_logistic (REFIT_TRAIN only);
  references (constant, time_only, p_only, PT_linear, PT_flex) from the row table alone; only
    p_only / PT_linear / PT_flex read the restricted column pt_p_pre.
``kind: calibrate`` fits the q calibrator of each listed result on its variant's Q_CAL rows.
The result (RESULT.json) carries the Q_DEV key hash shared by every cell, the release digest
and resource records.  TEST rows are never read: their labels are sealed in the row table.

CLI: python -B studies/q_production_20261005/q_worker.py --spec <run>/jobs/<dir>/SPEC.json
Exit codes: 0 completed, 1 failed (retryable), 3 failed terminally (not retried).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Callable, Mapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from studies.label_production_20261005.assemble import write_npz  # noqa: E402
from studies.q_production_20261005 import features as qf  # noqa: E402
from studies.q_production_20261005 import qbaselines, qbatch, qcalibrate, qmodels, qtrain, release, rows as qrows  # noqa: E402
from studies.v_model_fit_input_20261003.models import LogisticNonconvergenceError, PlainMLP, SharedSlotMLP, TeamSetMLP  # noqa: E402

VERSION = "q_worker_v1_20261005"
TRAIN, Q_DEV, Q_CAL = 0, 3, 4
REFERENCES = ("constant", "time_only", "p_only", "PT_linear", "PT_flex")
READS_P = ("p_only", "PT_linear", "PT_flex")
TABLE_BATCH = 8192          # rows per transform call when a table family stacks a whole split
EXIT = {"completed": 0, "failed": 1, "terminal": 3}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def keys_sha256(keys: np.ndarray) -> str:
    return hashlib.sha256(b"\n".join(bytes(k) for k in np.asarray(keys))).hexdigest()


def _checked(path: Path, expected: str, label: str) -> Path:
    if qrows.sha256_file(path) != expected:
        raise ValueError(f"{label} {path} differs from the spec")
    return path


def splits(rows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    role, stop = np.asarray(rows["role"]), np.asarray(rows["q_stop"], dtype=bool)
    train = np.flatnonzero(role == TRAIN)
    return {"train": train, "fit": train[~stop[train]], "stop": train[stop[train]],
            "q_dev": np.flatnonzero(role == Q_DEV), "q_cal": np.flatnonzero(role == Q_CAL)}


# ----------------------------------------------------------------------------- model structure
def tied_start_width(hidden: int) -> int:
    """nearest8((6H^2 + 9H) / (3H + 4)): one shared phi of width W has the parameters of G1's two phis."""

    return qmodels.nearest8((6 * hidden ** 2 + 9 * hidden) / (3 * hidden + 4))


def structure(family: str, card: Mapping[str, Any], context: qbatch.ScopeContext) -> dict[str, Any]:
    """Structural widths fixed for both phases from the REFIT_TRAIN encoder (EXECUTION_CARDS.json graph)."""

    if family not in ("team_set_pair_readout", "tied_message"):
        return {}
    hidden = int(card["hidden_width"])
    ek = context.encoder_kwargs(family)
    reference = qmodels.parameter_count(qmodels.QGraphModel("relational_mpnn", hidden_width=hidden, encoder_kwargs=ek))
    if family == "team_set_pair_readout":
        width, gap = qmodels.matched_width(hidden, lambda k: qmodels.QGraphModel(family, hidden_width=hidden, encoder_kwargs=ek, g0_width=k),
                                           reference, qmodels.g0_start_width(hidden))
        return {"g0_width": width, "start_width": qmodels.g0_start_width(hidden), "parameter_gap": gap, "g1_parameters": reference}
    width, gap = qmodels.matched_width(hidden, lambda w: qmodels.QGraphModel(family, hidden_width=hidden, encoder_kwargs=ek, tied_width=w),
                                       reference, tied_start_width(hidden))
    return {"tied_width": width, "start_width": tied_start_width(hidden), "parameter_gap": gap, "g1_parameters": reference}


def builder(family: str, card: Mapping[str, Any], context: qbatch.ScopeContext, cond: qbatch.Condition, shape: Mapping[str, Any],
            probe_rows: np.ndarray) -> Callable[[], Any]:
    ek = context.encoder_kwargs(family)
    if family in qmodels.GRAPH_FAMILIES:
        hidden = int(card["hidden_width"])
        return lambda: qmodels.QGraphModel(family, hidden_width=hidden, encoder_kwargs=ek, g0_width=shape.get("g0_width"),
                                           tied_width=shape.get("tied_width"), weight_decay=float(card.get("weight_decay", 0.0)))
    if family == "shared_slot_mlp":
        return lambda: SharedSlotMLP(**ek, include_side_flag=False, card=dict(card))
    if family == "pure_set":
        return lambda: TeamSetMLP(**ek, include_side_flag=False, card=dict(card))
    if family == "plain_mlp":
        width = int(qbatch.assemble(context, probe_rows[:1], cond, family).shape[1])
        return lambda: PlainMLP(input_dim=width, card=dict(card))
    if family == "ft_transformer":
        fields = qbatch.ft_fields(context)
        return lambda: qmodels.FTTransformer(**fields, d_token=int(card["d_token"]), layers=int(card["layers"]), heads=int(card["heads"]))
    raise ValueError(f"{family} is not a neural family")


# ----------------------------------------------------------------------------- references
def run_reference(job: Mapping[str, Any], rows: Mapping[str, np.ndarray], out: Path) -> dict[str, Any]:
    name, card = job["family"], dict(job["card"])
    part = splits(rows)
    train = part["train"]
    y = np.asarray(rows["y"], dtype=np.float64)
    w_raw = np.asarray(rows["weight_raw"], dtype=np.float64)
    t = np.asarray(rows["elapsed_min"], dtype=np.float64)
    # Allowlist: only the p-based references read the restricted pre probability (EXECUTION_CARDS q.references).
    p = np.asarray(rows["pt_p_pre"], dtype=np.float64) if name in READS_P else np.full(len(t), 0.5)
    if name == "constant":
        rate = qbaselines.constant_rate(y[train], w_raw[train])
        predict = lambda idx: np.full(len(idx), float(qbaselines.logit_p(np.asarray([rate]))[0]))  # noqa: E731
        model_path = out / "model_refit.json"
        qtrain.atomic_json(model_path, {"rate": rate, "rule": "TRAIN match-weighted positive rate"})
    else:
        try:
            model, design = qbaselines.fit_reference(name, card, {"p": p[train], "t": t[train], "y": y[train],
                                                                  "w_norm": qrows.normalized_weights(rows["match_index"][train])})
        except LogisticNonconvergenceError as error:
            return {"status": "failed", "reason": "nonconverged", "terminal": True, "selectable": False, "message": str(error),
                    "convergence": getattr(error, "convergence", None)}
        except ValueError as error:
            if "knots" not in str(error):
                raise
            return {"status": "failed", "reason": "degenerate_knots", "terminal": True, "selectable": False, "message": str(error)}
        model_path = out / "model_refit.pkl"
        with model_path.open("wb") as stream:
            pickle.dump({"model": model, "design": design}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        predict = lambda idx: np.asarray(model.predict_logits(design(p[idx], t[idx])), dtype=np.float64)  # noqa: E731
    scores = {}
    for split in ("q_dev", "q_cal"):
        idx = part[split]
        logits = predict(idx)
        sha = write_npz(out / f"pred_{split}.npz", {"row": idx, "logit": logits, "p": qtrain.sigmoid(logits)})
        scores[split] = {"rows": int(len(idx)), "raw_match_brier": qtrain.match_brier(qtrain.sigmoid(logits), y[idx], w_raw[idx]), "pred_sha256": sha}
    if name != "constant":
        with model_path.open("rb") as stream:
            again = pickle.load(stream)
        diff = float(np.abs(np.asarray(again["model"].predict_logits(again["design"](p[part["q_dev"]], t[part["q_dev"]])))
                            - np.load(out / "pred_q_dev.npz")["logit"]).max())
        if diff > 1e-7:
            raise RuntimeError(f"reference reload differs by {diff}")
    return {"status": "completed", "model_sha256": qtrain.sha256_file(model_path), "scores": scores,
            **({"rate": rate} if name == "constant" else {})}


# ----------------------------------------------------------------------------- model jobs
def run_model(spec: Mapping[str, Any], rows: Mapping[str, np.ndarray], out: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    job, inputs = spec["job"], spec["inputs"]
    family, card = job["family"], dict(job["card"])
    store = qf.RawStore(Path(inputs["store"]))
    xy = np.load(_checked(Path(inputs["xy"]), inputs["xy_sha256"], "XY"), allow_pickle=False)
    cond = qbatch.Condition.from_job(SimpleNamespace(factors=job["factors"], inputs=job["inputs"]))
    part = splits(rows)
    scopes = ("REFIT_TRAIN",) if family == "logistic" else ("FIT", "REFIT_TRAIN")
    ctx = {scope: qbatch.ScopeContext(store, rows, xy, scope, saved=Path(inputs["contexts"][scope])) for scope in scopes}
    for scope, context in ctx.items():
        if context.record.get("rows_sha256") != inputs["rows_sha256"] or context.record.get("xy_sha256") != inputs["xy_sha256"]:
            raise ValueError(f"{scope} context was prepared from other rows or XY")
    neural = family not in ("lightgbm", "logistic")
    rules = qtrain.neural_rules(family, card.get("weight_decay")) if neural else None
    batch = rules.batch_size if neural else TABLE_BATCH
    mi, w = np.asarray(rows["match_index"]), np.asarray(rows["weight_raw"], dtype=np.float64)

    def src(scope: str, idx: np.ndarray, weights: np.ndarray):
        return qbatch.source(ctx[scope], idx, cond, family, batch, weights)

    sources = {"refit": src("REFIT_TRAIN", part["train"], qrows.normalized_weights(mi[part["train"]])),
               "q_dev": src("REFIT_TRAIN", part["q_dev"], w[part["q_dev"]]),
               "q_cal": src("REFIT_TRAIN", part["q_cal"], w[part["q_cal"]])}
    if "FIT" in ctx:
        sources["fit"] = src("FIT", part["fit"], qrows.normalized_weights(mi[part["fit"]]))
        sources["stop"] = src("FIT", part["stop"], w[part["stop"]])
    if family == "logistic":
        return qtrain.fit_logistic(card, sources, out)
    if family == "lightgbm":
        return qtrain.fit_lightgbm(card, sources, out, identity)
    shape = structure(family, card, ctx["REFIT_TRAIN"])
    builds = {"FIT": builder(family, card, ctx["FIT"], cond, shape, part["fit"]),
              "REFIT": builder(family, card, ctx["REFIT_TRAIN"], cond, shape, part["train"])}
    import torch

    device = spec.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    qtrain.configure_runtime()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        result = qtrain.fit_neural(builds, qbatch.forward_for(family), sources, rules, out, identity, device)
    except qtrain.FitTimeLimit as error:
        return {"status": "failed", "reason": "fit_time_limit", "terminal": True, "selectable": False, "message": str(error)}
    result.update(structure=shape, rules=asdict(rules), device=device,
                  parameters={phase: qmodels.parameter_count(qtrain._new_model(build, torch.device("cpu"))) for phase, build in builds.items()})
    if device.startswith("cuda"):
        result["gpu"] = {"name": torch.cuda.get_device_name(0), "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    return result


# ----------------------------------------------------------------------------- calibration
def run_calibrate(spec: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """One calibrator per listed target: {"job_id", "result_dir", "rows", "rows_sha256", "constant"}.

    Calibrator files go to ``spec["calibrators_dir"]`` (default out/calibrators) and are written once:
    an existing file for the same Q_CAL predictions is kept, one for other predictions is an error."""

    records = {}
    tables: dict[str, dict[str, np.ndarray]] = {}
    target_dir = Path(spec.get("calibrators_dir") or out / "calibrators")
    for target in spec["targets"]:
        name = target["job_id"].replace("/", "__").replace(":", "~")
        path = target_dir / f"{name}.json"
        pred_sha = qtrain.sha256_file(Path(target["result_dir"]) / "pred_q_cal.npz")
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing["pred_q_cal_sha256"] != pred_sha:
                raise ValueError(f"{path} was fitted on other Q_CAL predictions")
            records[target["job_id"]] = {"status": existing["calibrator"]["status"], "existing": True}
            continue
        rows_file = target["rows"]
        if rows_file not in tables:
            tables[rows_file] = load_npz(_checked(Path(rows_file), target["rows_sha256"], "rows"))
        rows = tables[rows_file]
        pred = load_npz(Path(target["result_dir"]) / "pred_q_cal.npz")
        idx = np.asarray(pred["row"], dtype=np.int64)
        if not (np.asarray(rows["role"])[idx] == Q_CAL).all():
            raise ValueError(f"{target['job_id']}: calibration rows are not Q_CAL rows")
        fitted = qcalibrate.fit_q_calibrator(pred["p"], rows["y"][idx], rows["match_index"][idx], constant_model=bool(target.get("constant")))
        record = {"job_id": target["job_id"], "pred_q_cal_sha256": pred_sha, "rows_sha256": target["rows_sha256"], "calibrator": fitted.to_dict()}
        qtrain.atomic_json(path, _jsonable(record))
        records[target["job_id"]] = {"status": fitted.status, "reasons": list(fitted.reasons), "a": fitted.a, "b": fitted.b}
    return {"status": "completed", "calibrators": records, "fitted": sum(r["status"] == "fitted" for r in records.values())}


# ----------------------------------------------------------------------------- entry
def run(spec: Mapping[str, Any], out: Path) -> dict[str, Any]:
    if spec.get("kind") == "calibrate":
        return run_calibrate(spec, out)
    job, inputs = spec["job"], spec["inputs"]
    rows = load_npz(_checked(Path(inputs["rows"]), inputs["rows_sha256"], "rows"))
    identity = {"job_identity": job["identity"], "variant": spec["variant"], "rows_sha256": inputs["rows_sha256"], "release": spec["release_digest"]}
    if job["family"] in REFERENCES:
        result = run_reference(job, rows, out)
    else:
        result = run_model(spec, rows, out, identity)
    if result.get("status") == "completed":
        result["q_dev_keys_sha256"] = keys_sha256(np.asarray(rows["engagement_key"])[splits(rows)["q_dev"]])
    return result


def _jsonable(value: Any) -> Any:
    """Non-finite floats become strings so a failure record is always writable (atomic_json refuses NaN)."""

    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def _peak_rss() -> int | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or info.rss)
    except Exception:  # noqa: BLE001 - resource records are best effort
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out = Path(spec["out_dir"])
    result_path = out / "RESULT.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed" and previous.get("spec_sha256") == qtrain.sha256_file(spec_path):
            return EXIT["completed"]
    started = time.time()
    payload = release.verify(Path(spec["release"]))
    if payload["digest"] != spec["release_digest"]:
        raise RuntimeError("spec names another release digest")
    release.check_loaded(payload)
    try:
        result = run(spec, out)
    except Exception as error:  # noqa: BLE001 - recorded, then the supervisor decides on a retry
        result = {"status": "failed", "reason": type(error).__name__, "terminal": False, "message": str(error), "traceback": traceback.format_exc()}
    try:
        release.check_loaded(payload)
    except RuntimeError as error:
        result = {"status": "failed", "reason": "release_closure", "terminal": True, "message": str(error)}
    result.update(version=VERSION, job_id=spec.get("job", {}).get("job_id", spec.get("kind")), identity=spec.get("job", {}).get("identity"),
                  variant=spec.get("variant"), release_digest=payload["digest"], spec_sha256=qtrain.sha256_file(spec_path),
                  seconds=round(time.time() - started, 1), finished_utc=utc(), host=platform.node(), pid=os.getpid(), peak_rss_bytes=_peak_rss(),
                  threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    qtrain.atomic_json(result_path, _jsonable(result))
    if result["status"] == "completed":
        return EXIT["completed"]
    return EXIT["terminal"] if result.get("terminal") else EXIT["failed"]


if __name__ == "__main__":
    raise SystemExit(main())
