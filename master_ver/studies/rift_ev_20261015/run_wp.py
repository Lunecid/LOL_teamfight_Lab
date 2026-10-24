"""RIFT-EV under the paper's q protocol: main label 1[SVI > 0], every q row, the Table 3 family-selection rule.

Pre-registration: PLAN_WP.md (written before any fit).  As in q_run_1 (paper Table 3): rows and weights of
Q_ROWS_h90.npz unchanged, full input N_G1_S1_A1_M1_E1_C1_R1_D1, four cards, seed 20261001, the card with the
lowest Q_DEV raw match-weighted Brier is the family's row.  Added after that row: the selected RIFT card and
the q_run_1 LightGBM card (lgbm_l15_m200) at seeds 20261002 and 20261003, and a seed-20261001 LightGBM refit
that must reproduce the q_run_1 cell.  TEST labels stay sealed (qbatch.source refuses y < 0).

Subcommands: fit --kind rift|lgbm --card C --seed S | launch --stage search|seeds | select | evaluate
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from studies.rift_ev_20261015.run import (GRAPH_SPEC, LEVEL, RIFT_CARDS, RUN, SEEDS, TEST, cell_key, condition, contexts,  # noqa: E402
                                          lgbm_spec_path, read_json, rift_builder, rift_source, utc, write_once)

HERE = Path(__file__).resolve().parent
OUT = Path(r"D:\LOL_Training_20261004\rift_ev_2_wp")
VERSION = "rift_ev_wp_protocol_v1_20261015"
LABEL_TAG = "q_main_svi_sign"
LGBM_CARD = "lgbm_l15_m200"
EXPECTED_ROWS = {"train": 268101, "fit": 241529, "stop": 26572, "q_dev": 107970, "q_cal": 54061}
JOBS = RUN / "jobs"
FAMILIES = {   # paper Table 3 rows: q_run_1 selected cells (final cells reuse these search cells)
    "lightgbm": "search__auxiliary__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__lgbm_l15_m200",
    "graphsage": "search__auxiliary__SAGE_S1_A1_M1_E1_C1_R1_D1__graphsage__q_sage_h128_wd0.0001",
    "relational_mpnn": "search__common__N_G1_S1_A1_M1_E1_C1_R1_D1__relational_mpnn__q_shared_h64_wd0.0001",
    "team_set": "search__common__N_G0_S1_A1_M1_E1_C1_R1_D1__team_set_pair_readout__q_shared_h64_wd0.0001",
    "shared_slot_mlp": "search__auxiliary__FULL_LEGACY_SHARED_ENCODER_MLP_NEW_ADAPTER__shared_slot_mlp__q_slot_h64_wd0.0001",
    "plain_mlp": "search__auxiliary__FULL_PLAIN_MLP__plain_mlp__plain_256_128_wd0.001",
    "PT_flex": "search__reference__PT_FLEX__PT_flex__ptflex_k8_C0.001",
}


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("model.py", "run.py", "run_wp.py"):
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def cell_dir(key: str) -> Path:
    return OUT / "cells" / key


def load_rows(spec: dict) -> tuple[dict, dict[str, np.ndarray]]:
    from studies.q_production_20261005 import q_worker as W

    inputs = spec["inputs"]
    rows = W.load_npz(W._checked(Path(inputs["rows"]), inputs["rows_sha256"], "rows"))
    part = W.splits(rows)
    sizes = {name: int(len(idx)) for name, idx in part.items()}
    if sizes != EXPECTED_ROWS:
        raise RuntimeError(f"split sizes {sizes} differ from q_run_1 {EXPECTED_ROWS}")
    role = np.asarray(rows["role"])
    if any((role[idx] == TEST).any() for idx in part.values()):
        raise RuntimeError("a split contains TEST rows")
    return rows, part


def cmd_fit(kind: str, card_id: str, seed: int) -> int:
    import torch

    from studies.q_production_20261005 import q_worker as W
    from studies.q_production_20261005 import qbatch, qmodels, qtrain
    from studies.q_production_20261005 import rows as qrows
    from studies.rift_ev_20261015 import model as M

    key = cell_key(kind, card_id, seed)
    out = cell_dir(key)
    if (read_json(out / "RESULT.json") or {}).get("status") == "completed":
        return 0
    started = time.time()
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not pre-registered")
    if kind == "rift":
        card, spec_path = dict(RIFT_CARDS[card_id]), GRAPH_SPEC
    elif kind == "lgbm" and card_id == LGBM_CARD:
        spec_path = lgbm_spec_path(card_id)
        card = dict(read_json(spec_path)["job"]["card"])
    else:
        raise ValueError(f"{kind} {card_id} is not pre-registered")
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
                return qbatch.source(ctx[scope], idx, cond, "lightgbm", W.TABLE_BATCH, weights)
        sources = {"fit": src("FIT", part["fit"], qrows.normalized_weights(mi[part["fit"]])),
                   "stop": src("FIT", part["stop"], w[part["stop"]]),
                   "refit": src("REFIT_TRAIN", part["train"], qrows.normalized_weights(mi[part["train"]])),
                   "q_dev": src("REFIT_TRAIN", part["q_dev"], w[part["q_dev"]]),
                   "q_cal": src("REFIT_TRAIN", part["q_cal"], w[part["q_cal"]])}
        if kind == "rift":
            from dataclasses import asdict

            builds = {"FIT": rift_builder(ctx["FIT"], card), "REFIT": rift_builder(ctx["REFIT_TRAIN"], card)}
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                result = qtrain.fit_neural(builds, M.forward, sources, rules, out, identity, device)
            except qtrain.FitTimeLimit as error:
                result = {"status": "failed", "reason": "fit_time_limit", "message": str(error)}
            result.update(rules=asdict(rules), device=device,
                          parameters={phase: qmodels.parameter_count(qtrain._new_model(build, torch.device("cpu"))) for phase, build in builds.items()})
        else:
            result = qtrain.fit_lightgbm(card, sources, out, identity)
            if seed == SEEDS[0]:
                mine, theirs = np.load(out / "pred_q_dev.npz"), np.load(JOBS / FAMILIES["lightgbm"] / "pred_q_dev.npz")
                result["reproduces_q_run_1_lightgbm"] = {"rows_equal": bool(np.array_equal(mine["row"], theirs["row"])),
                                                         "max_abs_logit_diff": float(np.abs(mine["logit"] - theirs["logit"]).max())}
    except Exception as error:  # noqa: BLE001 - recorded
        result = {"status": "failed", "reason": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    result.update(version=VERSION, label=LABEL_TAG, key=key, identity=identity, seconds=round(time.time() - started, 1), finished_utc=utc(), pid=os.getpid())
    qtrain.atomic_json(out / "RESULT.json", W._jsonable(result))
    return 0 if result["status"] == "completed" else 1


def planned(stage: str) -> list[tuple[str, str, int]]:
    if stage == "search":
        return [("rift", c, SEEDS[0]) for c in RIFT_CARDS] + [("lgbm", LGBM_CARD, SEEDS[0])]
    selection = read_json(OUT / "SELECTION.json")
    if selection is None:
        raise RuntimeError("run select before the seed stage")
    return [("rift", selection["selected"]["rift"], s) for s in SEEDS[1:]] + [("lgbm", LGBM_CARD, s) for s in SEEDS[1:]]


def cmd_launch(stage: str, gpu_slots: int, cpu_slots: int) -> int:
    if not (OUT / "PREREG.json").exists():
        write_once(OUT / "PREREG.json", {"version": VERSION, "utc": utc(), "code_sha256": code_sha256(),
                                         "plan_sha256": hashlib.sha256((HERE / "PLAN_WP.md").read_bytes()).hexdigest(),
                                         "rift_cards": RIFT_CARDS, "lgbm_card": LGBM_CARD, "seeds": list(SEEDS), "label": LABEL_TAG,
                                         "expected_rows": EXPECTED_ROWS, "families": FAMILIES, "level": LEVEL})
    if read_json(OUT / "PREREG.json")["code_sha256"] != code_sha256():
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
            if proc.poll() is not None:
                del running[job]
                note(f"ended {cell_key(*job)} code={proc.returncode}")
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


def cmd_select() -> int:
    import json

    target = OUT / "SELECTION.json"
    if target.exists():
        raise FileExistsError(target)
    cards = list(RIFT_CARDS)
    table = {}
    for card in cards:
        result = read_json(cell_dir(cell_key("rift", card, SEEDS[0])) / "RESULT.json") or {}
        if result.get("status") != "completed":
            raise RuntimeError(f"rift {card} is not completed")
        table[card] = {"q_dev_raw_match_brier": result["scores"]["q_dev"]["raw_match_brier"], "best_epoch": result["best_epoch"]}
    selected = min(cards, key=lambda c: (table[c]["q_dev_raw_match_brier"], cards.index(c)))
    families = {name: read_json(JOBS / key / "RESULT.json")["scores"]["q_dev"]["raw_match_brier"] for name, key in FAMILIES.items()}
    write_once(target, {"version": VERSION, "utc": utc(), "rule": "lowest Q_DEV raw match-weighted Brier at seed 20261001; ties by card order (paper Table 3)",
                        "rift_q_dev": table, "selected": {"rift": selected, "lgbm": LGBM_CARD}, "q_run_1_family_q_dev": families})
    print(json.dumps({"selected": selected, "rift_q_dev": table, "q_run_1_family_q_dev": families}, indent=1))
    return 0


def cmd_evaluate() -> int:
    import json

    from studies.q_production_20261005 import qeval, qtest

    target = OUT / "REPORT.json"
    if target.exists():
        raise FileExistsError(target)
    selection = read_json(OUT / "SELECTION.json")
    rows, part = load_rows(read_json(GRAPH_SPEC))
    y_all, match_all = np.asarray(rows["y"], dtype=np.float64), np.asarray(rows["match_index"], dtype=np.int64)
    cells = {f"rift_s{s}": cell_dir(cell_key("rift", selection["selected"]["rift"], s)) for s in SEEDS}
    cells.update({f"lgbm_s{s}": cell_dir(cell_key("lgbm", LGBM_CARD, s)) for s in SEEDS})
    cells.update({f"rift_search_{c}": cell_dir(cell_key("rift", c, SEEDS[0])) for c in RIFT_CARDS})
    cells.update({f"q_run_1_{name}": JOBS / key for name, key in FAMILIES.items()})
    third = 1.0 / len(SEEDS)
    rift_mean = {f"rift_s{s}": third for s in SEEDS}
    contrasts = {"primary_rift_minus_lgbm_seedmean": {**rift_mean, **{f"lgbm_s{s}": -third for s in SEEDS}},
                 **{f"rift_minus_lgbm_s{s}": {f"rift_s{s}": 1.0, f"lgbm_s{s}": -1.0} for s in SEEDS},
                 "paper_row_rift_s1_minus_q_run_1_lightgbm": {f"rift_s{SEEDS[0]}": 1.0, "q_run_1_lightgbm": -1.0},
                 **{f"rift_seedmean_minus_q_run_1_{name}": {**rift_mean, f"q_run_1_{name}": -1.0} for name in FAMILIES if name != "lightgbm"}}
    report = {"version": VERSION, "utc": utc(), "label": LABEL_TAG, "selection": selection, "prereg": read_json(OUT / "PREREG.json"),
              "fits": {k: read_json(v / "RESULT.json") for k, v in cells.items() if k.startswith(("rift_s", "lgbm_s"))},
              "cells": {k: str(v) for k, v in cells.items()}, "contrasts": contrasts, "splits": {}}
    for split in ("q_dev", "q_cal"):
        expected = part[split]
        preds = {}
        for name, path in cells.items():
            data = qtest.load_npz(path / f"pred_{split}.npz")
            if not np.array_equal(np.asarray(data["row"], dtype=np.int64), expected):
                raise RuntimeError(f"{name} {split}: prediction rows are not the q rows")
            preds[name] = np.asarray(data["p"], dtype=np.float64)
        y, match = y_all[expected], match_all[expected]
        metrics = {k: {m: qeval.evaluate(y, p, match)["match_weighted"][m] for m in ("brier", "logloss", "auc", "calibration_slope")} for k, p in preds.items()}
        losses = {k: qeval.per_match_loss(y, preds[k], match) for k in preds}
        shared, aligned = qeval.align_cells(losses)
        boot = qeval.paired_match_bootstrap(shared, aligned, contrasts, levels={name: LEVEL for name in contrasts})
        report["splits"][split] = {"rows": int(len(expected)), "matches": int(len(np.unique(match))), "metrics": metrics, "bootstrap": boot}
    dev = report["splits"]["q_dev"]["metrics"]
    paper = {name: dev[f"q_run_1_{name}"]["brier"] for name in FAMILIES if name != "PT_flex"}
    paper["rift_ev"] = dev[f"rift_s{SEEDS[0]}"]["brier"]
    report["paper_protocol_dev_ranking"] = sorted(paper.items(), key=lambda kv: kv[1])
    entries = report["splits"]["q_cal"]["bootstrap"]["contrasts"]
    primary = entries["primary_rift_minus_lgbm_seedmean"]
    per_seed = [float(entries[f"rift_minus_lgbm_s{s}"]["point"]) for s in SEEDS]
    report["criterion"] = {"rule": "Q_CAL primary 95% upper bound < 0 and every per-seed point difference < 0",
                           "primary_point": primary["point"], "primary_interval": primary["interval"], "per_seed_points": per_seed,
                           "met": bool(primary["interval"][1] < 0 and all(v < 0 for v in per_seed))}
    report["code_sha256"] = code_sha256()
    write_once(target, report)
    print(json.dumps({"paper_protocol_dev_ranking": report["paper_protocol_dev_ranking"], "criterion": report["criterion"]}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("fit", "launch", "select", "evaluate"))
    ap.add_argument("--kind", choices=("rift", "lgbm"))
    ap.add_argument("--card")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stage", choices=("search", "seeds"), default="search")
    ap.add_argument("--gpu-slots", type=int, default=2)
    ap.add_argument("--cpu-slots", type=int, default=2)
    args = ap.parse_args()
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
