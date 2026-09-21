#!/usr/bin/env python3
"""Overnight SVI pipeline — concordance/quiet upgrade, 352 cross-target, transfer, Tier-B deep.

Runs while operator sleeps. Non-smoke, full budgets. Epistemic: exploratory follow-up.

Stages (skip with --skip-*):
  1. concordance   — decided-set denoms; agree+disagree=1; disagreement taxonomy
  2. quiet_match    — type-B with p_pre / clock / duration matching + signed ΔV
  3. cross_target   — LightGBM on 352 ridge for SVI / kill / obj; kill→SVI cal substitute
  4. transfer       — frozen q (iq EXT preds) + frozen V→W (v_EXT_*.npz)
  5. tier_b         — FT-Transformer + TabNet on 352 ridge (Track-A-like OPT, lean grids)

Writes outputs/svi_overnight_20260919/{status.jsonl, REPORT.md, stage_* /}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

ROLE = "EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"
OUT = REPO / "outputs" / "svi_overnight_20260919"


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "incremental_q_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def log(msg: str, out_dir: Path) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (out_dir / "status.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                            "msg": msg}, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ───────────────────────────────────────────────────────────────────── stage 1
def stage_concordance(data_root: Path, out_dir: Path) -> Dict[str, Any]:
    import rr20260919_svi_validation_suite as suite
    import rr20260919_svi_cohort_aligned as aligned

    splits = {name: suite.load_split_t(data_root, name) for name, _ in suite.SPLITS}
    combat = suite.combat_kill_alive(data_root, splits)
    # pool T
    keys = ["match", "s", "y", "delta", "p_pre", "obj"]
    sp = {k: np.concatenate([splits[n][k] for n, _ in suite.SPLITS]) for k in keys}
    kd = np.concatenate([combat[n]["kd"] for n, _ in suite.SPLITS])
    al = np.concatenate([combat[n]["al"] for n, _ in suite.SPLITS])
    y = sp["y"].astype(int)
    w = suite.match_weights(sp["match"])

    def decided_table(sign_arr: np.ndarray, label: str) -> Dict[str, Any]:
        s = np.asarray(sign_arr, float)
        finite = np.isfinite(s)
        tie = finite & (s == 0)
        decided = finite & (s != 0)
        n_all = int(finite.sum())
        n_dec = int(decided.sum())
        if n_dec == 0:
            return dict(label=label, n_all=n_all, n_decided=0, n_tie=int(tie.sum()),
                        tie_share_of_finite=1.0, agree=None, disagree=None)
        pred_pos = (s[decided] > 0)
        agree_arr = (pred_pos & (y[decided] == 1)) | ((~pred_pos) & (y[decided] == 0))
        ww = w[decided]
        agree = float(np.average(agree_arr, weights=ww))
        disagree = float(1.0 - agree)
        # unweighted counts
        n_agree = int(agree_arr.sum())
        n_disagree = int((~agree_arr).sum())
        return dict(
            label=label,
            weight="match_inverse_frequency on decided rows",
            n_all_finite=n_all,
            n_decided=n_dec,
            n_tie=int(tie.sum()),
            tie_share_of_finite=float(tie.sum() / max(1, n_all)),
            n_agree=n_agree,
            n_disagree=n_disagree,
            agree=agree,
            disagree=disagree,
            agree_plus_disagree=agree + disagree,
            note="agree+disagree must equal 1 on decided set (weighted)",
        )

    kill_s = np.sign(kd)
    obj_s = np.sign(sp["obj"].astype(float))
    alive_s = np.sign(al)
    tables = dict(
        kill=decided_table(kill_s, "kill_net"),
        objective=decided_table(obj_s, "objective_net"),
        alive=decided_table(alive_s, "alive_diff"),
    )

    # disagreement taxonomy among kill-decided disagree with SVI
    decided_k = np.isfinite(kd) & (kill_s != 0)
    kill_pos = kill_s > 0
    svi_pos = y == 1
    disagree_m = decided_k & ((kill_pos & ~svi_pos) | (~kill_pos & svi_pos))
    n_dis = int(disagree_m.sum())
    taxonomy = {}
    if n_dis:
        od = obj_s[disagree_m]
        yd = y[disagree_m]
        # obj with SVI / with kill / tie / other
        obj_with_svi = (od != 0) & (((od > 0) & (yd == 1)) | ((od < 0) & (yd == 0)))
        obj_with_kill = (od != 0) & (((od > 0) & kill_pos[disagree_m]) | ((od < 0) & ~kill_pos[disagree_m]))
        obj_tie = od == 0
        taxonomy = dict(
            n_kill_svi_disagree=n_dis,
            obj_with_SVI=dict(n=int(obj_with_svi.sum()), share=float(obj_with_svi.mean())),
            obj_with_kill=dict(n=int(obj_with_kill.sum()), share=float(obj_with_kill.mean())),
            obj_tie=dict(n=int(obj_tie.sum()), share=float(obj_tie.mean())),
            obj_other=dict(
                n=int((~(obj_with_svi | obj_with_kill | obj_tie)).sum()),
                share=float((~(obj_with_svi | obj_with_kill | obj_tie)).mean()),
            ),
            language="correspondence with observed material outcomes (not independent external truth)",
        )

    out = dict(
        epistemic=ROLE,
        population="pooled T 15.14+15.15+15.16 (measurement)",
        n_rows=int(len(y)),
        n_matches=int(len(np.unique(sp["match"]))),
        tables=tables,
        kill_svi_disagreement_taxonomy=taxonomy,
    )
    write_json(out_dir / "stage_concordance" / "results.json", out)
    return out


# ───────────────────────────────────────────────────────────────────── stage 2
def stage_quiet_match(data_root: Path, out_dir: Path, n_matches: int, seed: int) -> Dict[str, Any]:
    """Type-B quiet with post-hoc matching on p_pre, clock, duration vs fight rows."""
    import rr20260919_svi_validation_suite as suite
    import rr20260919_svi_cohort_aligned as aligned

    splits = {name: suite.load_split_t(data_root, name) for name, _ in suite.SPLITS}
    keys = ["match", "s", "y", "delta", "p_pre", "obj"]
    sp = {k: np.concatenate([splits[n][k] for n, _ in suite.SPLITS]) for k in keys}
    pick = aligned.sample_match_ids(sp["match"], n_matches, seed, "overnight_quiet")
    log(f"quiet: type-B on {len(pick)} matches…", out_dir)
    qb = aligned.quiet_type_b_on_matches(data_root, sp, pick)

    # Build fight row table for matching
    mid_set = set(qb.get("match_ids_loaded") or [])
    m = np.array([str(x) in mid_set for x in sp["match"]])
    fight = dict(
        p_pre=sp["p_pre"][m].astype(float),
        time_min=sp["s"][m].astype(float) / 60000.0,
        duration_min=np.full(m.sum(), 1.5),  # h90 = 90s
        delta=sp["delta"][m].astype(float),
        match=sp["match"][m].astype(str),
    )

    # Re-score quiet windows with covariates for matching (reload light)
    suite._setup_wt(data_root)
    import fc20260915_common as C  # noqa: E402
    from fc20260915_common import load_v_adapter  # noqa: E402
    from data.cache_io import load_match_cache  # noqa: E402
    from core.config import NODE_FEATURE_NAMES  # noqa: E402
    from gameplay.state_value_v2 import StateBuilder, state_matrix  # noqa: E402
    from gameplay.state_value import final_outcome  # noqa: E402

    adapter = load_v_adapter(C.OUT / "models" / "v" / "v_final_raw.joblib")
    names = list(adapter.state_names)
    H_MS = 90_000
    q_rows = []
    for mid in (qb.get("match_ids_loaded") or []):
        pack = load_match_cache(str(mid))
        if pack is None:
            continue
        try:
            _, terminal = final_outcome(pack["events"])
            builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
        except Exception:
            continue
        kills = np.asarray(
            [int(e["timestamp"]) for e in pack.get("events", [])
             if str(e.get("type", e.get("eventType", ""))).upper() == "CHAMPION_KILL"],
            dtype=np.int64)
        ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        if len(ts) < 3:
            continue
        last = int(min(terminal if terminal > 0 else ts[-1], ts[-1]))
        n_chosen = 0
        for t0 in range(120_000, max(120_000, last - H_MS), 30_000):
            t1 = t0 + H_MS
            if t1 > last:
                break
            if kills.size and np.any((kills >= t0) & (kills <= t1)):
                continue
            try:
                s0, s1 = builder.at(t0), builder.at(t0 + H_MS - 1)
                X = state_matrix([s0, s1], names)
                p = adapter.predict_matrix(X, names, adapter.state_version)
            except Exception:
                continue
            q_rows.append(dict(
                match=str(mid),
                p_pre=float(p[0]),
                time_min=float(t0) / 60000.0,
                duration_min=1.5,
                delta=float(p[1] - p[0]),
            ))
            n_chosen += 1
            if n_chosen >= 3:
                break

    if not q_rows or fight["delta"].size == 0:
        out = dict(status="partial", quiet_type_b_unmatched=qb, matched=None)
        write_json(out_dir / "stage_quiet" / "results.json", out)
        return out

    qp = np.array([r["p_pre"] for r in q_rows])
    qt = np.array([r["time_min"] for r in q_rows])
    qd = np.array([r["delta"] for r in q_rows])
    # bin match: p_pre 0.1, time 5 min
    def bin_key(p, t):
        return (int(np.clip(p, 0, 0.999) * 10), int(np.clip(t, 0, 59) // 5))

    fight_bins: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(fight["delta"])):
        fight_bins.setdefault(bin_key(fight["p_pre"][i], fight["time_min"][i]), []).append(i)
    quiet_bins: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(qd)):
        quiet_bins.setdefault(bin_key(qp[i], qt[i]), []).append(i)

    paired_f, paired_q = [], []
    rng = np.random.default_rng(seed)
    for k, f_ix in fight_bins.items():
        q_ix = quiet_bins.get(k) or []
        if not q_ix:
            continue
        n = min(len(f_ix), len(q_ix))
        f_s = rng.choice(f_ix, size=n, replace=False)
        q_s = rng.choice(q_ix, size=n, replace=False)
        paired_f.extend(f_s.tolist())
        paired_q.extend(q_s.tolist())

    fd = fight["delta"][np.array(paired_f, dtype=int)] if paired_f else np.array([])
    qq = qd[np.array(paired_q, dtype=int)] if paired_q else np.array([])

    def sumstats(d: np.ndarray) -> Dict[str, Any]:
        if len(d) == 0:
            return dict(n=0)
        return dict(
            n=int(len(d)),
            mean_abs=float(np.abs(d).mean()),
            median_abs=float(np.median(np.abs(d))),
            mean_signed=float(d.mean()),
            P_pos=float((d > 0).mean()),
        )

    matched = dict(
        design="type_B_matched_on_p_pre_decile_x_time5min_same_duration_h90",
        n_pairs=int(len(fd)),
        fight=sumstats(fd),
        quiet=sumstats(qq),
        mean_abs_ratio=(float(np.abs(fd).mean() / max(1e-9, np.abs(qq).mean())) if len(fd) else None),
        median_abs_ratio=(float(np.median(np.abs(fd)) / max(1e-9, np.median(np.abs(qq)))) if len(fd) else None),
        mean_signed_fight_minus_quiet=(float(fd.mean() - qq.mean()) if len(fd) else None),
        note="Not pure fight contribution; matched no-kill reference under similar start state/time/length",
    )
    out = dict(
        epistemic=ROLE,
        n_matches_requested=n_matches,
        quiet_type_b_unmatched=dict(
            n_scored=qb.get("n_scored"),
            abs_ratio=qb.get("abs_ratio"),
            quiet_mean_abs=qb.get("quiet_mean_abs"),
            quiet_P_pos=qb.get("quiet_P_pos"),
            fight=qb.get("fight"),
        ),
        matched_state_time_duration=matched,
    )
    write_json(out_dir / "stage_quiet" / "results.json", out)
    return out


# ───────────────────────────────────────────────────────────────────── stage 3
def stage_cross_target(data_root: Path, out_dir: Path) -> Dict[str, Any]:
    """352-ridge LightGBM for SVI/kill/obj; kill→SVI calibrated substitute on 15.16."""
    sys.path.insert(0, str(data_root / "scripts"))
    import fc20260915_common as C  # noqa: E402
    import cr20260915_common as K  # noqa: E402
    import iq20260915_common as Q  # noqa: E402
    import lightgbm as lgb
    import rr20260919_svi_validation_suite as suite

    schema = C.read_json(Q.FC / "q_pre_only_schema.json")
    ridge = list(schema["predictor_sets"]["ridge"])
    D = Q.load_trainval(Q.OUT, smoke=False, cohort="T")
    names = D["names"]
    cols = [names.index(n) for n in ridge]
    X = D["X"][:, cols]
    y_svi = D["y"].astype(int)
    g = D["g"].astype(str)
    M = {k: D["role"] == k for k in ("TRAIN", "Q_CAL", "Q_SELECT")}

    # combat for TRAIN+TEST
    tr = suite.load_split_t(data_root, "MAIN_TRAIN")
    te = suite.load_split_t(data_root, "MAIN_TEST")
    combat = suite.combat_kill_alive(data_root, {"MAIN_TRAIN": tr, "MAIN_TEST": te})

    # Map trainval rows to kill/obj via (match,s) — use labels during counts for obj; kill from combat
    # Simpler: fit on MAIN_TRAIN / MAIN_TEST splits with ridge features from parent sets
    def load_xy(set_name: str, target: str):
        F, Lb, Co, _ = Q.load_parent_set(set_name, Q.OUT, f"cross_target {set_name}")
        m = (Lb["valid_h90"] == 1) & (Co["cohort"] == Q.COHORT_CODE["T"])
        names_f = list(F["input_names"]) if "input_names" in F else list(schema["input_names_all"])
        Xr = F["X_input"][m][:, [names_f.index(n) for n in ridge]]
        g_ = F["match"][m].astype(str)
        y_s = Lb["Y_h90"][m].astype(int)
        if target == "svi":
            return Xr, y_s, g_, np.ones(int(m.sum()), dtype=bool)
        # join combat
        sp = tr if set_name == "MAIN_TRAIN" else te
        cmb = combat["MAIN_TRAIN" if set_name == "MAIN_TRAIN" else "MAIN_TEST"]
        key_sp = {k: i for i, k in enumerate(sp["key"].tolist())}
        keys = np.char.add(F["match"][m].astype(str), np.char.add("|", F["s_ms"][m].astype(str)))
        ix = np.array([key_sp.get(k, -1) for k in keys])
        ok = ix >= 0
        if target == "kill":
            kd = cmb["kd"][ix[ok]]
            sk = np.sign(kd)
            keep = np.isfinite(kd) & (sk != 0)
            y = (sk[keep] > 0).astype(int)
            return Xr[ok][keep], y, g_[ok][keep], keep
        if target == "obj":
            obj = sp["obj"][ix[ok]]
            so = np.sign(obj)
            keep = so != 0
            y = (so[keep] > 0).astype(int)
            return Xr[ok][keep], y, g_[ok][keep], keep
        raise ValueError(target)

    def fit_lgbm(Xtr, ytr, gtr, Xte, yte, gte):
        wtr = suite.match_weights(gtr)
        wte = suite.match_weights(gte)
        dtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        # sealed winner-like config
        params = dict(
            objective="binary", metric="binary_logloss", learning_rate=0.05,
            num_leaves=15, min_child_samples=100, feature_fraction=0.9,
            bagging_fraction=0.9, bagging_freq=1, verbosity=-1, seed=7,
        )
        booster = lgb.train(params, dtrain, num_boost_round=400)
        pred = booster.predict(Xte)
        br = float(np.average((pred - yte) ** 2, weights=wte))
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(yte, pred, sample_weight=wte))
        except ValueError:
            auc = None
        return booster, dict(brier=br, auc=auc, n=int(len(yte)), n_matches=int(len(np.unique(gte)))), pred

    results = {}
    boosters = {}
    preds = {}
    for target in ("svi", "kill", "obj"):
        log(f"cross_target fit {target}…", out_dir)
        Xtr, ytr, gtr, _ = load_xy("MAIN_TRAIN", target)
        Xte, yte, gte, _ = load_xy("MAIN_TEST", target)
        booster, met, pred = fit_lgbm(Xtr, ytr, gtr, Xte, yte, gte)
        results[target] = met
        boosters[target] = booster
        preds[target] = dict(y=yte, g=gte, p=pred)

    # kill→SVI substitute: calibrate kill scores on Q_CAL-like MAIN_VALIDATION to SVI, eval on TEST
    log("kill→SVI calibrated substitute…", out_dir)
    Xtr_k, ytr_k, gtr_k, _ = load_xy("MAIN_TRAIN", "kill")
    # train kill model already have; score SVI rows on TEST with kill model then calibrate
    # Fit calibrator: kill-model raw → SVI on VALIDATION decided? Use MAIN_VALIDATION as cal
    Xva, yva_svi, gva, _ = load_xy("MAIN_VALIDATION", "svi")
    # For kill model applied to all SVI rows (not only kill-decided)
    Xtr_s, _, gtr_s, _ = load_xy("MAIN_TRAIN", "svi")
    ytr_s = load_xy("MAIN_TRAIN", "svi")[1]
    # Refit kill model is wrong for substitute — use kill booster to score SVI-eval rows
    Xte_s, yte_s, gte_s, _ = load_xy("MAIN_TEST", "svi")
    raw_te = boosters["kill"].predict(Xte_s)
    raw_va = boosters["kill"].predict(Xva)
    cals = K.fit_calibrators(raw_va, yva_svi, suite.match_weights(gva))
    sub = {}
    for cal in K.CALS:
        if cal == "raw":
            p = raw_te
        else:
            p = K.apply_calibration(cal, cals[cal], raw_te)
        wte = suite.match_weights(gte_s)
        br = float(np.average((p - yte_s) ** 2, weights=wte))
        sub[cal] = dict(brier=br, n=int(len(yte_s)))
    best_cal = min(sub.keys(), key=lambda c: (sub[c]["brier"], c))
    # compare to native SVI model
    substitute = dict(
        kill_scores_calibrated_to_SVI=sub,
        best_cal=best_cal,
        native_svi_brier=results["svi"]["brier"],
        delta_brier_sub_minus_native=sub[best_cal]["brier"] - results["svi"]["brier"],
        note="Positive delta ⇒ kill-substitute worse than SVI-native on SVI label",
        cal_fit="MAIN_VALIDATION T; eval MAIN_TEST T; kill model TRAIN fit",
    )

    out = dict(
        epistemic=ROLE,
        n_inputs=352,
        family="LightGBM L15_M100-like",
        targets=results,
        kill_to_svi_substitute=substitute,
        eval_sample="MAIN_TEST T decided rows per target",
    )
    write_json(out_dir / "stage_cross_target" / "results.json", out)
    return out


# ───────────────────────────────────────────────────────────────────── stage 4
def stage_transfer(data_root: Path, out_dir: Path) -> Dict[str, Any]:
    import rr20260919_svi_transfer_2026 as tr

    iq = data_root / "outputs" / "incremental_q_training_20260915"
    q_block = {}
    for set_name, label, role, pilot in tr.COHORTS:
        path = iq / "eval" / "predictions" / f"{set_name}_h90_T.npz"
        if not path.is_file():
            q_block[set_name] = dict(error="missing", path=str(path))
            continue
        q_block[set_name] = dict(label=label, role=role, pilot=pilot, **tr.metrics_from_npz(path))

    # V → W from v_EXT_*.npz
    v_dir = data_root / "outputs" / "full_corpus_training_20260915" / "eval" / "predictions"
    v_block = {}
    for set_name, label, role, pilot in tr.COHORTS:
        path = v_dir / f"v_{set_name}.npz"
        if not path.is_file():
            v_block[set_name] = dict(error="missing", path=str(path))
            continue
        z = np.load(path, allow_pickle=False)
        y = z["winner_blue"].astype(int)
        # unique match: take one row per match (first query) for match-level; also all bucket samples
        g = z["match"].astype(str)
        _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
        w = (1.0 / c[inv]).astype(np.float64)
        p = z["p_sigmoid_pos"].astype(float) if "p_sigmoid_pos" in z.files else z["p_raw"].astype(float)
        br = float(np.average((p - y) ** 2, weights=w))
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y, p, sample_weight=w))
        except ValueError:
            auc = None
        # time bands via query_ms
        tmin = z["query_ms"].astype(float) / 60000.0
        bands = {}
        for lo, hi, name in ((0, 10, "0_10"), (10, 20, "10_20"), (20, 30, "20_30"), (30, 1e9, "30_inf")):
            m = (tmin >= lo) & (tmin < hi)
            if m.sum() < 50:
                continue
            ww = w[m]
            bands[name] = dict(
                n=int(m.sum()),
                brier=float(np.average((p[m] - y[m]) ** 2, weights=ww)),
                auc=float(roc_auc_score(y[m], p[m], sample_weight=ww)),
            )
        v_block[set_name] = dict(
            label=label, role=role, pilot=pilot,
            n=int(len(y)), n_matches=int(len(np.unique(g))),
            brier=br, auc=auc, time_bands=bands,
            frozen="v_final from full_corpus; score-only",
        )

    out = dict(
        epistemic=ROLE,
        freeze="ˆV, q, PT, b(p), preproc, calibrators — no refit on external",
        q_to_SVI=q_block,
        V_to_W=v_block,
    )
    write_json(out_dir / "stage_transfer" / "results.json", out)
    return out


# ───────────────────────────────────────────────────────────────────── stage 5
def stage_tier_b(data_root: Path, out_dir: Path, device: str) -> Dict[str, Any]:
    """FT-Transformer + TabNet lean grids on 352 ridge (same protocol as TabM)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    sys.path.insert(0, str(data_root / "scripts"))
    import fc20260915_common as C  # noqa: E402
    import cr20260915_common as K  # noqa: E402
    import iq20260915_common as Q  # noqa: E402

    schema = C.read_json(Q.FC / "q_pre_only_schema.json")
    ridge = list(schema["predictor_sets"]["ridge"])
    D = Q.load_trainval(Q.OUT, smoke=False, cohort="T")
    cols = [D["names"].index(n) for n in ridge]
    X_all, y_all, g_all = D["X"][:, cols], D["y"].astype(int), D["g"].astype(str)
    M = {k: D["role"] == k for k in ("TRAIN", "Q_CAL", "Q_SELECT")}

    class FTTiny(nn.Module):
        def __init__(self, n_in, d_token=64, depth=2, n_heads=4, dropout=0.1):
            super().__init__()
            self.n_in = n_in
            self.tok = nn.Linear(1, d_token)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
                dropout=dropout, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=depth)
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

        def forward(self, x):
            # x: (B, n_in)
            t = self.tok(x.unsqueeze(-1))  # (B, n_in, d)
            cls = self.cls.expand(x.size(0), -1, -1)
            h = self.enc(torch.cat([cls, t], dim=1))
            return self.head(h[:, 0])

    class TabNetTiny(nn.Module):
        def __init__(self, n_in, n_d=32, n_steps=3, dropout=0.1):
            super().__init__()
            self.n_steps = n_steps
            self.bn = nn.BatchNorm1d(n_in)
            self.shared = nn.Linear(n_in, n_d * 2)
            self.steps = nn.ModuleList([nn.Linear(n_d * 2, n_d * 2) for _ in range(n_steps)])
            self.att = nn.ModuleList([nn.Linear(n_d, n_in) for _ in range(n_steps)])
            self.out = nn.Linear(n_d, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            x = self.bn(x)
            prior = torch.ones_like(x)
            out = 0
            for i in range(self.n_steps):
                h = Fnn.relu(self.shared(x * prior))
                h = Fnn.relu(self.steps[i](h))
                d, a = h[:, : h.size(1) // 2], h[:, h.size(1) // 2 :]
                mask = torch.softmax(self.att[i](a), dim=-1)
                prior = prior * (1.0 - mask)
                out = out + self.drop(d)
            return self.out(out)

    SEEDS = (7, 42, 123)
    OPT = dict(lr=1e-3, weight_decay=1e-4, batch_size=512, max_epochs=80, patience=10)
    FT_GRID = (
        dict(d_token=64, depth=2),
        dict(d_token=128, depth=2),
        dict(d_token=128, depth=3),
    )
    TABNET_GRID = (
        dict(n_d=16, n_steps=3),
        dict(n_d=32, n_steps=3),
        dict(n_d=32, n_steps=5),
    )

    def pre(X):
        imp = SimpleImputer(strategy="median").fit(X)
        sc = StandardScaler().fit(imp.transform(X))
        return imp, sc

    def train_family(family: str, grid: Tuple[dict, ...]) -> Dict[str, Any]:
        Xtr, ytr, gtr = X_all[M["TRAIN"]], y_all[M["TRAIN"]], g_all[M["TRAIN"]]
        stop = Q.stop_mask(gtr)
        fit90 = ~stop
        imp90, sc90 = pre(Xtr[fit90])
        Z90 = sc90.transform(imp90.transform(Xtr[fit90])).astype(np.float32)
        Zst = sc90.transform(imp90.transform(Xtr[stop])).astype(np.float32)
        imp, sc = pre(Xtr)
        Zall = sc.transform(imp.transform(Xtr)).astype(np.float32)
        w_fit = Q.weights(gtr[fit90]).astype(np.float32)
        w_stop = Q.weights(gtr[stop]).astype(np.float32)
        w_all = Q.weights(gtr).astype(np.float32)

        cand_scores = {}
        best_overall = None

        for cfg in grid:
            name = (f"ft_d{cfg['d_token']}_L{cfg['depth']}" if family == "ft"
                    else f"tabnet_d{cfg['n_d']}_S{cfg['n_steps']}")
            log(f"tier_b {family} {name}…", out_dir)
            seed_probs_cal = []
            seed_probs_sel = []
            seed_probs_te = []  # filled after sealed
            states = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                if family == "ft":
                    model = FTTiny(Zall.shape[1], **cfg).to(device)
                else:
                    model = TabNetTiny(Zall.shape[1], **cfg).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
                # stop phase
                best_b, best_ep, wait = 1e9, 0, 0
                Zg = torch.tensor(Z90, device=device)
                yg = torch.tensor(ytr[fit90].astype(np.float32), device=device)
                wg = torch.tensor(w_fit, device=device)
                Zs = torch.tensor(Zst, device=device)
                ys = torch.tensor(ytr[stop].astype(np.float32), device=device)
                ws = torch.tensor(w_stop, device=device)
                gen = torch.Generator(device="cpu").manual_seed(seed)
                n, bs = len(ytr[fit90]), OPT["batch_size"]
                for ep in range(1, OPT["max_epochs"] + 1):
                    model.train()
                    perm = torch.randperm(n, generator=gen).to(device)
                    for i in range(0, n, bs):
                        idx = perm[i : i + bs]
                        logits = model(Zg[idx]).squeeze(1)
                        loss = (wg[idx] * Fnn.binary_cross_entropy_with_logits(logits, yg[idx], reduction="none")).mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        p = torch.sigmoid(model(Zs).squeeze(1)).cpu().numpy()
                    b = float(np.average((p - ytr[stop]) ** 2, weights=w_stop))
                    if b < best_b:
                        best_b, best_ep, wait = b, ep, 0
                    else:
                        wait += 1
                    if wait >= OPT["patience"]:
                        break
                # refit on all TRAIN for best_ep epochs
                torch.manual_seed(seed)
                if family == "ft":
                    model = FTTiny(Zall.shape[1], **cfg).to(device)
                else:
                    model = TabNetTiny(Zall.shape[1], **cfg).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
                Zg = torch.tensor(Zall, device=device)
                yg = torch.tensor(ytr.astype(np.float32), device=device)
                wg = torch.tensor(w_all, device=device)
                gen = torch.Generator(device="cpu").manual_seed(seed)
                n = len(ytr)
                for ep in range(1, max(1, best_ep) + 1):
                    model.train()
                    perm = torch.randperm(n, generator=gen).to(device)
                    for i in range(0, n, bs):
                        idx = perm[i : i + bs]
                        logits = model(Zg[idx]).squeeze(1)
                        loss = (wg[idx] * Fnn.binary_cross_entropy_with_logits(logits, yg[idx], reduction="none")).mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                model.eval()
                states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

                def predict(Znp):
                    model.eval()
                    with torch.no_grad():
                        outp = []
                        Zt = torch.tensor(Znp, device=device)
                        for i in range(0, len(Znp), 8192):
                            outp.append(torch.sigmoid(model(Zt[i : i + 8192]).squeeze(1)).cpu().numpy())
                    return np.concatenate(outp)

                Zcal = sc.transform(imp.transform(X_all[M["Q_CAL"]])).astype(np.float32)
                Zsel = sc.transform(imp.transform(X_all[M["Q_SELECT"]])).astype(np.float32)
                seed_probs_cal.append(predict(Zcal))
                seed_probs_sel.append(predict(Zsel))

            raw_cal = np.mean(seed_probs_cal, axis=0)
            raw_sel = np.mean(seed_probs_sel, axis=0)
            cals = K.fit_calibrators(raw_cal, y_all[M["Q_CAL"]], Q.weights(g_all[M["Q_CAL"]]))
            for cal in K.CALS:
                cand = f"{name}__{cal}"
                if cal == "raw":
                    psel = raw_sel
                else:
                    psel = K.apply_calibration(cal, cals[cal], raw_sel)
                br = float(np.average((psel - y_all[M["Q_SELECT"]]) ** 2,
                                      weights=Q.weights(g_all[M["Q_SELECT"]])))
                cand_scores[cand] = dict(
                    brier=br,
                    config=cfg,
                    cal=cal,
                    name=name,
                    family=family,
                    imp=imp, sc=sc, states=states, cals=cals,
                )

        # pick best by Q_SELECT brier
        ranking = sorted(cand_scores.keys(), key=lambda c: (cand_scores[c]["brier"], c))
        chosen = ranking[0]
        ch = cand_scores[chosen]
        # sealed MAIN_TEST
        Feat, Lb, Co, _ = Q.load_parent_set("MAIN_TEST", Q.OUT, "tier_b sealed")
        m = (Lb["valid_h90"] == 1) & (Co["cohort"] == Q.COHORT_CODE["T"])
        names_f = list(Feat["input_names"]) if "input_names" in Feat else list(schema["input_names_all"])
        Xte = Feat["X_input"][m][:, [names_f.index(n) for n in ridge]]
        yte = Lb["Y_h90"][m].astype(int)
        gte = Feat["match"][m].astype(str)
        Zte = ch["sc"].transform(ch["imp"].transform(Xte)).astype(np.float32)
        # rebuild model & average seeds
        preds = []
        for st in ch["states"]:
            if family == "ft":
                model = FTTiny(Zte.shape[1], **ch["config"]).to(device)
            else:
                model = TabNetTiny(Zte.shape[1], **ch["config"]).to(device)
            model.load_state_dict(st)
            model.eval()
            with torch.no_grad():
                outp = []
                Zt = torch.tensor(Zte, device=device)
                for i in range(0, len(Zte), 8192):
                    outp.append(torch.sigmoid(model(Zt[i : i + 8192]).squeeze(1)).cpu().numpy())
            preds.append(np.concatenate(outp))
        raw = np.mean(preds, axis=0)
        if ch["cal"] == "raw":
            p = raw
        else:
            p = K.apply_calibration(ch["cal"], ch["cals"][ch["cal"]], raw)
        wte = Q.weights(gte)
        brier = float(np.average((p - yte) ** 2, weights=wte))
        try:
            auc = float(roc_auc_score(yte, p, sample_weight=wte))
        except ValueError:
            auc = None
        # vs PT
        iq_pred = Q.OUT / "eval" / "predictions" / "MAIN_TEST_h90_T.npz"
        z = np.load(iq_pred, allow_pickle=False)
        keys_te = list(zip(gte.tolist(), Feat["s_ms"][m].astype(np.int64).tolist()))
        keys_z = list(zip(z["match"].astype(str).tolist(), z["s_ms"].astype(np.int64).tolist()))
        idx = {k: i for i, k in enumerate(keys_z)}
        order = np.array([idx[k] for k in keys_te])
        pt = z["named__pt_winner"][order]
        lgbm = z["named__lgbm_winner"][order]
        return dict(
            chosen=chosen,
            q_select_brier=ch["brier"],
            ranking=ranking[:6],
            select_top={c: cand_scores[c]["brier"] for c in ranking[:6]},
            sealed=dict(
                n=int(len(yte)),
                brier=brier,
                auc=auc,
                delta_vs_pt=float(brier - np.average((pt - yte) ** 2, weights=wte)),
                delta_vs_lgbm=float(brier - np.average((lgbm - yte) ** 2, weights=wte)),
            ),
            grid_size=len(grid),
            seeds=list(SEEDS),
            optimizer=OPT,
        )

    # FT with 352 tokens is memory-heavy — smaller batch
    try:
        # temporarily shrink batch for FT via monkeypatching OPT in train_family is awkward;
        # run FT with reduced grid depth if OOM — catch and continue
        ft = train_family("ft", FT_GRID)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            log(f"FT OOM at batch 512; retry batch 128…", out_dir)
            OPT["batch_size"] = 128
            try:
                ft = train_family("ft", (FT_GRID[0],))  # leaner
            except Exception as e2:
                ft = dict(error=str(e2), traceback=traceback.format_exc()[-2000:])
        else:
            ft = dict(error=str(e), traceback=traceback.format_exc()[-2000:])
    except Exception as e:
        ft = dict(error=str(e), traceback=traceback.format_exc()[-2000:])
        log(f"FT failed: {e}", out_dir)

    try:
        tn = train_family("tabnet", TABNET_GRID)
    except Exception as e:
        tn = dict(error=str(e), traceback=traceback.format_exc()[-2000:])
        log(f"TabNet failed: {e}", out_dir)

    out = dict(epistemic=ROLE, FT_Transformer=ft, TabNet=tn,
               note="SAINT deferred (row-attention inference contract). Limited search budget.")
    write_json(out_dir / "stage_tier_b" / "results.json", out)
    return out


def write_master_report(out_dir: Path, stages: Dict[str, Any]) -> None:
    lines = [
        "# SVI overnight pipeline",
        "",
        f"Generated: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"**Epistemic:** {ROLE}",
        "",
    ]
    c = stages.get("concordance") or {}
    if c:
        lines += ["## 1. Concordance (pooled T)", ""]
        for name, t in (c.get("tables") or {}).items():
            lines.append(
                f"- **{name}**: decided n={t.get('n_decided')}, agree={t.get('agree')}, "
                f"disagree={t.get('disagree')}, sum={t.get('agree_plus_disagree')}, "
                f"tie_share={t.get('tie_share_of_finite')}"
            )
        tax = c.get("kill_svi_disagreement_taxonomy") or {}
        if tax:
            lines.append(f"- Kill–SVI disagree n={tax.get('n_kill_svi_disagree')}; "
                         f"obj⇄SVI={tax.get('obj_with_SVI')}; obj tie={tax.get('obj_tie')}")
        lines.append("")
    q = stages.get("quiet") or {}
    if q:
        m = q.get("matched_state_time_duration") or {}
        lines += ["## 2. Quiet (matched p_pre × time × duration)", ""]
        lines.append(f"- pairs={m.get('n_pairs')}, mean|ΔV| ratio={m.get('mean_abs_ratio')}, "
                     f"median ratio={m.get('median_abs_ratio')}, "
                     f"signed fight−quiet={m.get('mean_signed_fight_minus_quiet')}")
        lines.append("")
    x = stages.get("cross_target") or {}
    if x:
        lines += ["## 3. Cross-target (352 LightGBM)", ""]
        for t, met in (x.get("targets") or {}).items():
            lines.append(f"- {t}: Brier={met.get('brier')} AUC={met.get('auc')} n={met.get('n')}")
        sub = x.get("kill_to_svi_substitute") or {}
        lines.append(f"- kill→SVI best ({sub.get('best_cal')}): "
                     f"ΔBrier vs native={sub.get('delta_brier_sub_minus_native')}")
        lines.append("")
    t = stages.get("transfer") or {}
    if t:
        lines += ["## 4. Transfer", ""]
        lines.append("### q → SVI")
        for k, v in (t.get("q_to_SVI") or {}).items():
            if "error" in v:
                lines.append(f"- {k}: {v['error']}")
            else:
                lines.append(f"- {k}: LGBM−PT ΔBrier={v.get('lgbm_minus_pt_brier')} "
                             f"(n={v.get('n')})")
        lines.append("### V → W")
        for k, v in (t.get("V_to_W") or {}).items():
            if "error" in v:
                lines.append(f"- {k}: {v['error']}")
            else:
                lines.append(f"- {k}: AUC={v.get('auc')} Brier={v.get('brier')} n={v.get('n')}")
        lines.append("")
    b = stages.get("tier_b") or {}
    if b:
        lines += ["## 5. Tier B (FT / TabNet)", ""]
        for fam in ("FT_Transformer", "TabNet"):
            block = b.get(fam) or {}
            if "error" in block:
                lines.append(f"- {fam}: ERROR {block['error']}")
            else:
                s = block.get("sealed") or {}
                lines.append(f"- {fam} chosen `{block.get('chosen')}`: "
                             f"sealed Brier={s.get('brier')} vs PT={s.get('delta_vs_pt')} "
                             f"vs LGBM={s.get('delta_vs_lgbm')}")
        lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-matches-quiet", type=int, default=2500)
    ap.add_argument("--skip-concordance", action="store_true")
    ap.add_argument("--skip-quiet", action="store_true")
    ap.add_argument("--skip-cross-target", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--skip-tier-b", action="store_true")
    args = ap.parse_args(argv)

    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"START overnight data_root={data_root} device={args.device}", out_dir)

    stages: Dict[str, Any] = {}
    t0 = time.time()

    def run(name, fn, skip):
        if skip:
            log(f"SKIP {name}", out_dir)
            stages[name] = dict(skipped=True)
            return
        log(f"BEGIN {name}", out_dir)
        t1 = time.time()
        try:
            stages[name] = fn()
            log(f"END {name} ({time.time() - t1:.1f}s)", out_dir)
        except Exception as e:
            log(f"FAIL {name}: {e}", out_dir)
            stages[name] = dict(error=str(e), traceback=traceback.format_exc()[-4000:])
            write_json(out_dir / f"stage_{name}" / "error.json", stages[name])

    run("concordance", lambda: stage_concordance(data_root, out_dir), args.skip_concordance)
    run("quiet", lambda: stage_quiet_match(data_root, out_dir, args.n_matches_quiet, 7), args.skip_quiet)
    run("cross_target", lambda: stage_cross_target(data_root, out_dir), args.skip_cross_target)
    run("transfer", lambda: stage_transfer(data_root, out_dir), args.skip_transfer)
    run("tier_b", lambda: stage_tier_b(data_root, out_dir, args.device), args.skip_tier_b)

    write_master_report(out_dir, stages)
    summary = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        seconds=round(time.time() - t0, 1),
        stages={k: ("ok" if "error" not in (v or {}) and not (v or {}).get("skipped")
                    else ("skipped" if (v or {}).get("skipped") else "error"))
                for k, v in stages.items()},
        epistemic=ROLE,
    )
    write_json(out_dir / "results.json", summary)
    log(f"DONE total={time.time() - t0:.1f}s → {out_dir}", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
