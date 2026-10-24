"""RIFT-EV Phase 1: snapshot RIFT-EV against LightGBM on the kill-advantage label, q inputs and q rules.

Pre-registration: PLAN.md (written before any fit).  Rows, label and weights are the q_ka_1 ones (KA_v1:
eligible rows KA != 0, raw weights 1/m over eligible rows); inputs are the q_run_1 condition
N_G1_S1_A1_M1_E1_C1_R1_D1 (FIT and REFIT_TRAIN scope contexts, read only); training follows qtrain
(FIT with STOP early stopping, REFIT on all TRAIN for the best epoch or round count, score Q_DEV and Q_CAL).
TEST rows are never assembled.  Locked q_run_1 / q_ka_1 artifacts are read only; outputs go to OUT once.

Subcommands:
  smoke                               shapes, edge layout, determinism, parameter counts, batch timing (no writes)
  fit --kind rift|lgbm --card C --seed S
  launch --stage search|seeds [--gpu-slots N --cpu-slots N]
  select                              Q_DEV only: selected card per kind -> SELECTION.json
  evaluate                            Q_DEV and Q_CAL metrics and paired match bootstrap -> REPORT.json
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = Path(r"D:\LOL_Training_20261004\q_run_1")
KA = Path(r"D:\LOL_Training_20261004\q_ka_1")
OUT = Path(r"D:\LOL_Training_20261004\rift_ev_1")
VERSION = "rift_ev_phase1_v1_20261015"
LABEL_TAG = "KA_v1"
TRAIN, Q_DEV, Q_CAL, TEST = 0, 3, 4, 5
GRAPH_SPEC = RUN / "jobs" / "search__common__N_G1_S1_A1_M1_E1_C1_R1_D1__relational_mpnn__q_shared_h128_wd0.0001" / "SPEC.json"
KA_P1 = KA / "cells" / "final__module__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__selected~lightgbm"
KA_PT = KA / "cells" / "final__reference__PT_FLEX__PT_flex__selected~PT_flex"
ARCH = {"layers": 3, "heads": 4, "queries": 4, "experts": 3, "ffn_mult": 2}
RIFT_CARDS = {f"rift_d{d}_wd{wd}": {"width": d, "weight_decay": wd, **ARCH} for d in (64, 128) for wd in (0.0001, 0.001)}
LGBM_CARDS = ("lgbm_l15_m100", "lgbm_l15_m200", "lgbm_l31_m100", "lgbm_l31_m200")
SEEDS = (20261001, 20261002, 20261003)
EXPECTED_ROWS = {"train": 241756, "fit": 217781, "stop": 23975, "q_dev": 97434, "q_cal": 48944}
LEVEL = 0.95


def lgbm_spec_path(card: str) -> Path:
    return RUN / "jobs" / f"search__auxiliary__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__{card}" / "SPEC.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_once(path: Path, obj) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("model.py", "run.py"):
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def cell_key(kind: str, card: str, seed: int) -> str:
    return f"{kind}__{card}__s{seed}"


def cell_dir(key: str) -> Path:
    return OUT / "cells" / key


# ----------------------------------------------------------------------------- data
def load_rows(spec: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """q row table with y and weight_raw replaced by KA_v1, and the eligible split indices."""

    from studies.q_production_20261005 import q_worker as W
    from studies.q_production_20261005 import qtest

    inputs = spec["inputs"]
    base = W.load_npz(W._checked(Path(inputs["rows"]), inputs["rows_sha256"], "rows"))
    prepare = read_json(KA / "KA_PREPARE.json")
    if qtest.sha256_file(KA / "KA_LABELS.npz") != prepare["ka_labels_sha256"] or prepare["rows_sha256"] != inputs["rows_sha256"]:
        raise RuntimeError("KA labels do not verify against KA_PREPARE.json")
    ka = qtest.load_npz(KA / "KA_LABELS.npz")
    if not np.array_equal(ka["engagement_key"], base["engagement_key"]):
        raise RuntimeError("KA labels are not aligned with the q row table")
    rows = dict(base)
    rows["y"] = np.asarray(ka["y_ka"], dtype=np.float64)
    rows["weight_raw"] = np.asarray(ka["weight_raw_ka"], dtype=np.float64)
    eligible = np.asarray(ka["eligible"], dtype=bool)
    part = {name: idx[eligible[idx]] for name, idx in W.splits(rows).items()}
    sizes = {name: int(len(idx)) for name, idx in part.items()}
    if sizes != EXPECTED_ROWS:
        raise RuntimeError(f"eligible split sizes {sizes} differ from q_ka_1 {EXPECTED_ROWS}")
    role = np.asarray(rows["role"])
    if any((role[idx] == TEST).any() for idx in part.values()):
        raise RuntimeError("a split contains TEST rows")
    return rows, part


def contexts(spec: dict, rows: dict) -> dict:
    from studies.q_production_20261005 import features as qf
    from studies.q_production_20261005 import q_worker as W
    from studies.q_production_20261005 import qbatch

    inputs = spec["inputs"]
    store = qf.RawStore(Path(inputs["store"]))
    xy = np.load(W._checked(Path(inputs["xy"]), inputs["xy_sha256"], "XY"), allow_pickle=False)
    ctx = {scope: qbatch.ScopeContext(store, rows, xy, scope, saved=Path(inputs["contexts"][scope])) for scope in ("FIT", "REFIT_TRAIN")}
    for scope, context in ctx.items():
        if context.record.get("rows_sha256") != inputs["rows_sha256"] or context.record.get("xy_sha256") != inputs["xy_sha256"]:
            raise ValueError(f"{scope} context was prepared from other rows or XY")
    return ctx


def condition(spec: dict):
    from studies.q_production_20261005 import qbatch

    job = spec["job"]
    return qbatch.Condition.from_job(SimpleNamespace(factors=job["factors"], inputs=job["inputs"]))


def rift_source(context, idx: np.ndarray, cond, batch_size: int, weights: np.ndarray):
    """qbatch.source for the graph inputs plus the raw pre-frame age (seconds) per row."""

    from studies.q_production_20261005 import qbatch, qtrain

    idx = np.asarray(idx, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if (np.asarray(context.rows["role"])[idx] == TEST).any():
        raise ValueError("a source would read TEST rows")
    y = np.asarray(context.rows["y"], dtype=np.float32)
    w_raw = np.asarray(context.rows["weight_raw"], dtype=np.float64)
    age = np.asarray(context.rows["frame_age_s"], dtype=np.float32)

    def make():
        for start in range(0, len(idx), batch_size):
            part = idx[start:start + batch_size]
            structured, graph = qbatch.assemble(context, part, cond, "relational_mpnn")
            yield qtrain.Batch((structured, graph, age[part]), y[part], weights[start:start + len(part)], w_raw[part], part)

    return lambda: qbatch._prefetch(make())


def rift_builder(context, card: dict):
    from studies.rift_ev_20261015.model import RiftEVSnapshot

    ek = context.encoder_kwargs("relational_mpnn")
    arch = {k: card[k] for k in ("width", *ARCH)}
    return lambda: RiftEVSnapshot(encoder_kwargs=ek, **arch)


# ----------------------------------------------------------------------------- smoke
def cmd_smoke() -> int:
    import torch

    from studies.q_production_20261005 import qbatch, qmodels, qtrain
    from studies.q_production_20261005 import rows as qrows
    from studies.rift_ev_20261015 import model as M

    t0 = time.time()
    spec = read_json(GRAPH_SPEC)
    rows, part = load_rows(spec)
    ctx = contexts(spec, rows)
    cond = condition(spec)
    print("rows", {k: len(v) for k, v in part.items()}, "load s", round(time.time() - t0, 1))
    qtrain.configure_runtime(SEEDS[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    idx = part["fit"][:1024]
    t1 = time.time()
    structured, graph = qbatch.assemble(ctx["FIT"], idx[:512], cond, "relational_mpnn")
    print("assemble 512 rows s", round(time.time() - t1, 3))
    xy = np.asarray(ctx["FIT"].xy[idx[:512]], dtype=np.float64)
    dist = np.sqrt(((xy[:, None, :, :] - xy[:, :, None, :]) ** 2).sum(-1)) / ctx["FIT"].sigma
    dense = M.dense_pairs(torch.as_tensor(graph.edge))[..., 2].numpy()
    print("dense edge layout max abs diff", float(np.abs(dense - dist).max()))
    ally_dense = M.dense_pairs(torch.as_tensor(graph.ally)).numpy()
    team = np.asarray(graph.team_ids)
    expected_ally = (team[:, :, None] == team[:, None, :]) & ~np.eye(10, dtype=bool)
    print("dense ally layout equal", bool(np.array_equal(ally_dense, expected_ally)))
    for name, card in RIFT_CARDS.items():
        torch.manual_seed(SEEDS[0])
        model = rift_builder(ctx["FIT"], card)()
        print(name, "parameters", qmodels.parameter_count(model))
    torch.manual_seed(SEEDS[0])
    reference = qmodels.QGraphModel("relational_mpnn", hidden_width=128, encoder_kwargs=ctx["FIT"].encoder_kwargs("relational_mpnn"))
    print("relational_mpnn h128 parameters", qmodels.parameter_count(reference))
    card = RIFT_CARDS["rift_d128_wd0.0001"]
    torch.manual_seed(SEEDS[0])
    model = rift_builder(ctx["FIT"], card)().to(device)
    rules = qtrain.neural_rules("rift_ev", card["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=rules.lr, weight_decay=rules.weight_decay)
    mi = np.asarray(rows["match_index"])
    source = rift_source(ctx["FIT"], idx, cond, rules.batch_size, qrows.normalized_weights(mi[idx]))
    model.train()
    for step, batch in enumerate(source()):
        t2 = time.time()
        logits = M.forward(model, batch.inputs, device)
        loss = torch.mean(torch.as_tensor(batch.w_norm, dtype=torch.float32, device=device)
                          * torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(batch.y, device=device), reduction="none"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), rules.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        print("train step", step, "loss", round(float(loss), 5), "s", round(time.time() - t2, 3), "finite", bool(torch.isfinite(logits).all()))
    _, a, _, _ = qtrain.predict(model, source, M.forward, device)
    _, b, _, _ = qtrain.predict(model, source, M.forward, device)
    print("eval determinism max abs diff", float(np.abs(a - b).max()), "p range", float(qtrain.sigmoid(a).min()), float(qtrain.sigmoid(a).max()))
    t3 = time.time()
    epoch_rows = 0
    for batch in rift_source(ctx["FIT"], part["fit"][:512 * 20], cond, 512, np.ones(512 * 20))():
        epoch_rows += len(batch.y)
        logits = M.forward(model, batch.inputs, device)
        loss = logits.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_batch = (time.time() - t3) / 20
    print("20 prefetch batches s/batch", round(per_batch, 3), "-> est. FIT epoch min", round(per_batch * len(part["fit"]) / 512 / 60, 1))
    print("code_sha256", code_sha256())
    return 0


# ----------------------------------------------------------------------------- fit
def cmd_fit(kind: str, card_id: str, seed: int) -> int:
    import torch

    from studies.q_production_20261005 import q_worker as W
    from studies.q_production_20261005 import qbatch, qmodels, qtrain
    from studies.q_production_20261005 import rows as qrows
    from studies.rift_ev_20261015 import model as M

    key = cell_key(kind, card_id, seed)
    out = cell_dir(key)
    result_path = out / "RESULT.json"
    if (read_json(result_path) or {}).get("status") == "completed":
        return 0
    started = time.time()
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not pre-registered")
    if kind == "rift":
        card, spec_path = dict(RIFT_CARDS[card_id]), GRAPH_SPEC
    elif kind == "lgbm":
        if card_id not in LGBM_CARDS:
            raise ValueError(f"{card_id} is not a pre-registered LightGBM card")
        spec_path = lgbm_spec_path(card_id)
        card = dict(read_json(spec_path)["job"]["card"])
    else:
        raise ValueError(kind)
    spec = read_json(spec_path)
    identity = {"version": VERSION, "kind": kind, "card_id": card_id, "card": card, "seed": seed, "label": LABEL_TAG,
                "rows_sha256": spec["inputs"]["rows_sha256"], "xy_sha256": spec["inputs"]["xy_sha256"], "code_sha256": code_sha256()}
    out.mkdir(parents=True, exist_ok=True)
    qtrain.SEED = seed          # read at call time by _new_model and fit_lightgbm
    qtrain.configure_runtime(seed)
    try:
        rows, part = load_rows(spec)
        ctx = contexts(spec, rows)
        cond = condition(spec)
        mi, w = np.asarray(rows["match_index"]), np.asarray(rows["weight_raw"], dtype=np.float64)
        if kind == "rift":
            rules = qtrain.neural_rules("rift_ev", card["weight_decay"])

            def src(scope, idx, weights):
                return rift_source(ctx[scope], idx, cond, rules.batch_size, weights)
        else:
            def src(scope, idx, weights):
                if (np.asarray(rows["role"])[idx] == TEST).any():
                    raise ValueError("a source would read TEST rows")
                return qbatch.source(ctx[scope], idx, cond, "lightgbm", W.TABLE_BATCH, weights)
        sources = {"fit": src("FIT", part["fit"], qrows.normalized_weights(mi[part["fit"]])),
                   "stop": src("FIT", part["stop"], w[part["stop"]]),
                   "refit": src("REFIT_TRAIN", part["train"], qrows.normalized_weights(mi[part["train"]])),
                   "q_dev": src("REFIT_TRAIN", part["q_dev"], w[part["q_dev"]]),
                   "q_cal": src("REFIT_TRAIN", part["q_cal"], w[part["q_cal"]])}
        if kind == "rift":
            builds = {"FIT": rift_builder(ctx["FIT"], card), "REFIT": rift_builder(ctx["REFIT_TRAIN"], card)}
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                result = qtrain.fit_neural(builds, M.forward, sources, rules, out, identity, device)
            except qtrain.FitTimeLimit as error:
                result = {"status": "failed", "reason": "fit_time_limit", "message": str(error)}
            from dataclasses import asdict
            result.update(rules=asdict(rules), device=device,
                          parameters={phase: qmodels.parameter_count(qtrain._new_model(build, torch.device("cpu"))) for phase, build in builds.items()})
            if device.startswith("cuda"):
                result["gpu"] = {"name": torch.cuda.get_device_name(0), "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated())}
        else:
            result = qtrain.fit_lightgbm(card, sources, out, identity)
            if card_id == "lgbm_l15_m200" and seed == SEEDS[0]:
                mine, theirs = np.load(out / "pred_q_dev.npz"), np.load(KA_P1 / "pred_q_dev.npz")
                result["reproduces_q_ka_1_P1"] = {"rows_equal": bool(np.array_equal(mine["row"], theirs["row"])),
                                                  "max_abs_logit_diff": float(np.abs(mine["logit"] - theirs["logit"]).max())}
    except Exception as error:  # noqa: BLE001 - recorded
        result = {"status": "failed", "reason": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    result.update(version=VERSION, label=LABEL_TAG, key=key, identity=identity, seconds=round(time.time() - started, 1), finished_utc=utc(), pid=os.getpid())
    qtrain.atomic_json(result_path, W._jsonable(result))
    return 0 if result["status"] == "completed" else 1


# ----------------------------------------------------------------------------- launch
def planned(stage: str) -> list[tuple[str, str, int]]:
    if stage == "search":
        return [("rift", c, SEEDS[0]) for c in RIFT_CARDS] + [("lgbm", c, SEEDS[0]) for c in LGBM_CARDS]
    selection = read_json(OUT / "SELECTION.json")
    if selection is None:
        raise RuntimeError("run select before the seed stage")
    return [(kind, selection["selected"][kind], s) for kind in ("rift", "lgbm") for s in SEEDS[1:]]


def cmd_launch(stage: str, gpu_slots: int, cpu_slots: int) -> int:
    if not (OUT / "PREREG.json").exists():
        write_once(OUT / "PREREG.json", {"version": VERSION, "utc": utc(), "code_sha256": code_sha256(), "plan_sha256": hashlib.sha256((HERE / "PLAN.md").read_bytes()).hexdigest(),
                                         "rift_cards": RIFT_CARDS, "lgbm_cards": list(LGBM_CARDS), "seeds": list(SEEDS), "label": LABEL_TAG,
                                         "expected_rows": EXPECTED_ROWS, "level": LEVEL})
    prereg = read_json(OUT / "PREREG.json")
    if prereg["code_sha256"] != code_sha256():
        raise RuntimeError("code changed after pre-registration")
    pending = [job for job in planned(stage) if (read_json(cell_dir(cell_key(*job)) / "RESULT.json") or {}).get("status") != "completed"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE",
           **{name: "2" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    slots = {"rift": gpu_slots, "lgbm": cpu_slots}
    running: dict[tuple, subprocess.Popen] = {}
    log = OUT / "LAUNCH.log"

    def note(text: str) -> None:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"{utc()} {text}\n")

    note(f"launch stage={stage} pending={pending}")
    while pending or running:
        for job, proc in list(running.items()):
            code = proc.poll()
            if code is not None:
                del running[job]
                note(f"ended {cell_key(*job)} code={code}")
        for job in list(pending):
            if sum(1 for r in running if r[0] == job[0]) < slots[job[0]]:
                pending.remove(job)
                key = cell_key(*job)
                cell_dir(key).mkdir(parents=True, exist_ok=True)
                with (cell_dir(key) / "worker.log").open("ab") as wl:
                    running[job] = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "fit", "--kind", job[0], "--card", job[1], "--seed", str(job[2])],
                                                    cwd=ROOT, stdout=wl, stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                note(f"started {key} pid={running[job].pid}")
        time.sleep(20)
    note(f"launch stage={stage} done")
    return 0


# ----------------------------------------------------------------------------- select (Q_DEV only)
def cmd_select() -> int:
    target = OUT / "SELECTION.json"
    if target.exists():
        raise FileExistsError(target)
    table, selected = {}, {}
    for kind, cards in (("rift", list(RIFT_CARDS)), ("lgbm", list(LGBM_CARDS))):
        table[kind] = {}
        for card in cards:
            result = read_json(cell_dir(cell_key(kind, card, SEEDS[0])) / "RESULT.json") or {}
            if result.get("status") != "completed":
                raise RuntimeError(f"{kind} {card} is not completed")
            table[kind][card] = {"q_dev_raw_match_brier": result["scores"]["q_dev"]["raw_match_brier"],
                                 **({"best_epoch": result["best_epoch"]} if kind == "rift" else {"best_iteration": result["best_iteration"]})}
        selected[kind] = min(cards, key=lambda c: (table[kind][c]["q_dev_raw_match_brier"], cards.index(c)))
    write_once(target, {"version": VERSION, "utc": utc(), "rule": "lowest Q_DEV KA raw match-weighted Brier at seed 20261001; ties by card order",
                        "q_dev": table, "selected": selected})
    print(json.dumps({"selected": selected, "q_dev": table}, indent=1))
    return 0


# ----------------------------------------------------------------------------- evaluate
def cmd_evaluate() -> int:
    from studies.q_production_20261005 import qeval, qtest

    target = OUT / "REPORT.json"
    if target.exists():
        raise FileExistsError(target)
    selection = read_json(OUT / "SELECTION.json")
    spec = read_json(GRAPH_SPEC)
    rows, part = load_rows(spec)
    y_all, match_all = np.asarray(rows["y"], dtype=np.float64), np.asarray(rows["match_index"], dtype=np.int64)
    cells = {f"{kind}_s{s}": cell_dir(cell_key(kind, selection["selected"][kind], s)) for kind in ("rift", "lgbm") for s in SEEDS}
    cells.update({f"rift_search_{c}": cell_dir(cell_key("rift", c, SEEDS[0])) for c in RIFT_CARDS})
    cells.update({f"lgbm_search_{c}": cell_dir(cell_key("lgbm", c, SEEDS[0])) for c in LGBM_CARDS})
    cells.update({"q_ka_1_P1_lgbm": KA_P1, "q_ka_1_PT_flex": KA_PT})
    third = 1.0 / len(SEEDS)
    contrasts = {"primary_rift_minus_lgbm_seedmean": {**{f"rift_s{s}": third for s in SEEDS}, **{f"lgbm_s{s}": -third for s in SEEDS}},
                 **{f"rift_minus_lgbm_s{s}": {f"rift_s{s}": 1.0, f"lgbm_s{s}": -1.0} for s in SEEDS},
                 "rift_seedmean_minus_q_ka_1_P1": {**{f"rift_s{s}": third for s in SEEDS}, "q_ka_1_P1_lgbm": -1.0},
                 "rift_seedmean_minus_PT_flex": {**{f"rift_s{s}": third for s in SEEDS}, "q_ka_1_PT_flex": -1.0},
                 "lgbm_seedmean_minus_PT_flex": {**{f"lgbm_s{s}": third for s in SEEDS}, "q_ka_1_PT_flex": -1.0}}
    report = {"version": VERSION, "utc": utc(), "label": LABEL_TAG, "selection": selection, "prereg": read_json(OUT / "PREREG.json"),
              "cells": {k: str(v) for k, v in cells.items()}, "contrasts": contrasts, "splits": {}}
    for split in ("q_dev", "q_cal"):
        expected = part[split]
        preds = {}
        for name, path in cells.items():
            data = qtest.load_npz(path / f"pred_{split}.npz")
            if not np.array_equal(np.asarray(data["row"], dtype=np.int64), expected):
                raise RuntimeError(f"{name} {split}: prediction rows are not the eligible rows")
            preds[name] = np.asarray(data["p"], dtype=np.float64)
        y, match = y_all[expected], match_all[expected]
        metrics = {k: {m: qeval.evaluate(y, p, match)["match_weighted"][m] for m in ("brier", "logloss", "auc", "calibration_slope")} for k, p in preds.items()}
        losses = {k: qeval.per_match_loss(y, preds[k], match) for k in preds}
        shared, aligned = qeval.align_cells(losses)
        boot = qeval.paired_match_bootstrap(shared, aligned, contrasts, levels={name: LEVEL for name in contrasts})
        report["splits"][split] = {"rows": int(len(expected)), "matches": int(len(np.unique(match))), "metrics": metrics, "bootstrap": boot}
    entries = report["splits"]["q_cal"]["bootstrap"]["contrasts"]
    primary = entries["primary_rift_minus_lgbm_seedmean"]
    per_seed = [float(entries[f"rift_minus_lgbm_s{s}"]["point"]) for s in SEEDS]
    report["criterion"] = {"rule": "Q_CAL primary 95% upper bound < 0 and every per-seed point difference < 0",
                           "primary_point": primary["point"], "primary_interval": primary["interval"], "per_seed_points": per_seed,
                           "met": bool(primary["interval"][1] < 0 and all(v < 0 for v in per_seed))}
    report["code_sha256"] = code_sha256()
    write_once(target, report)
    print(json.dumps({"criterion": report["criterion"], "q_cal_metrics": report["splits"]["q_cal"]["metrics"]}, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("smoke", "fit", "launch", "select", "evaluate"))
    ap.add_argument("--kind", choices=("rift", "lgbm"))
    ap.add_argument("--card")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stage", choices=("search", "seeds"), default="search")
    ap.add_argument("--gpu-slots", type=int, default=2)
    ap.add_argument("--cpu-slots", type=int, default=2)
    args = ap.parse_args()
    if args.command == "smoke":
        return cmd_smoke()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "fit":
        return cmd_fit(args.kind, args.card, args.seed)
    if args.command == "launch":
        return cmd_launch(args.stage, args.gpu_slots, args.cpu_slots)
    if args.command == "select":
        return cmd_select()
    return cmd_evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
