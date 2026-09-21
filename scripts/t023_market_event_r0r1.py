#!/usr/bin/env python3
"""E5 — market_event same-window correspondence + R0/R1 nextobj models.

Task: .ai/tasks/T023.md
Uses CACHE_MAIN. Frozen V for ΔV; TRAIN OOF ΔV for R0/R1 training labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BUNDLE = REPO / "outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator.joblib"
PRED_T = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
LAB = REPO / "outputs/q_newv_fit85_20260920/labels"
NEXTOBJ_TEST = REPO / "outputs/review_response_rr5_rr6b_20260920/next_objective_labels.npz"
DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E5_MARKET_R0R1_20260921.md"
NEXT_OBJ_WINDOW_MS = 180_000
ELIGIBLE = ("BARON_NASHOR", "DRAGON", "RIFTHERALD", "HORDE", "ATAKHAN")


def data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    for x in Path.home().iterdir():
        cand = x / "LOL_Teamfight"
        if (cand / "outputs" / "full_corpus_training_20260915").is_dir():
            return cand
    raise SystemExit("data root not found")


def setup(root: Path) -> None:
    wt = root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ["LOL_OUTPUT_ROOT"] = str(root / "outputs" / "full_corpus_training_20260915" / "runtime")


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def brier_binary(y, p, g) -> float:
    w = match_weights(g)
    return float(np.sum(w * (np.asarray(p, float) - np.asarray(y, float)) ** 2) / np.sum(w))


def game_end_ms(pack: dict) -> Optional[int]:
    info = pack.get("info") or {}
    if "gameDuration" in info and info["gameDuration"] is not None:
        v = int(info["gameDuration"])
        return v * 1000 if v < 10_000 else v
    for e in pack.get("events") or []:
        if e.get("type") == "GAME_END":
            return int(e.get("timestamp", 0) or 0)
    mts = pack.get("minute_ts")
    return int(mts[-1]) if mts is not None and len(mts) else None


def build_team_map(pack: dict) -> Dict[int, int]:
    tm = {}
    for p in pack.get("participants") or (pack.get("info") or {}).get("participants") or []:
        pid = int(p.get("participantId", 0) or 0)
        team = int(p.get("teamId", 0) or 0)
        if pid and team in (100, 200):
            tm[pid] = team
    if not tm and "meta" in pack and "team_map" in pack["meta"]:
        tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
    return tm


def classify_elite(e: dict) -> Optional[str]:
    if str(e.get("type", "")) != "ELITE_MONSTER_KILL":
        return None
    mt = str(e.get("monsterType", "") or "")
    return mt if mt in ELIGIBLE else None


def next_objective_label(events, team_map, endpoint, game_end) -> str:
    t1 = endpoint + NEXT_OBJ_WINDOW_MS
    if game_end is not None and game_end <= endpoint:
        return "game_ended_before_objective"
    cands = []
    for e in events:
        ts = int(e.get("timestamp", -1) or -1)
        if not (endpoint < ts <= t1):
            continue
        if classify_elite(e) is None:
            continue
        team = int(e.get("killerTeamId", 0) or 0)
        if team not in (100, 200):
            kid = int(e.get("killerId", 0) or 0)
            team = int(team_map.get(kid, 0))
        cands.append((ts, team))
    if not cands:
        if game_end is None:
            return "observation_censored"
        if endpoint < game_end < t1:
            return "game_ended_in_window_no_objective"
        return "none"
    cands.sort()
    t0 = cands[0][0]
    teams = {c[1] for c in cands if c[0] == t0}
    if len(teams) != 1 or 0 in teams:
        return "tie_ambiguous"
    team = next(iter(teams))
    return "Blue" if team == 100 else "Red"


def slot_diff(X, names, field):
    ix = {n: i for i, n in enumerate(names)}
    blue = [ix[f"participant_slot{s}_{field}"] for s in range(0, 5)]
    red = [ix[f"participant_slot{s}_{field}"] for s in range(5, 10)]
    return X[:, blue].sum(axis=1) - X[:, red].sum(axis=1)


def count_net(during, after, keys, blue, red):
    ki = {str(k): i for i, k in enumerate(keys)}
    C = during + after
    return C[:, ki[blue]].astype(float) - C[:, ki[red]].astype(float)


def material_row(E, i, names):
    keys = [str(k) for k in E["count_keys"]]
    during = E["during"][i : i + 1]
    after = E["after_h90"][i : i + 1]

    def net(b, r):
        return float(count_net(during, after, keys, b, r)[0])

    epic = (
        net("baron_blue", "baron_red")
        + net("dragon_blue", "dragon_red")
        + net("elder_blue", "elder_red")
        + net("herald_blue", "herald_red")
        + net("horde_blue", "horde_red")
        + net("atakhan_blue", "atakhan_red")
    )
    struct = net("tower_blue", "tower_red") + net("inhibitor_blue", "inhibitor_red")
    kill = float(
        slot_diff(E["X_post_h90"][i : i + 1], names, "kills")[0]
        - slot_diff(E["X_pre"][i : i + 1], names, "kills")[0]
    )
    return dict(kill_diff=kill, objective_net=epic + struct, epic_net=epic, structure_net=struct)


def multiclass_brier(y_idx: np.ndarray, proba: np.ndarray, g: np.ndarray, n_class: int) -> float:
    """Match-weighted sum of squared errors over one-hot classes."""
    w = match_weights(g)
    oh = np.zeros((len(y_idx), n_class), dtype=float)
    for i, yi in enumerate(y_idx):
        if 0 <= yi < n_class:
            oh[i, yi] = 1.0
    se = np.sum((proba - oh) ** 2, axis=1)
    return float(np.sum(w * se) / np.sum(w))


def main() -> int:
    root = data_root()
    setup(root)
    import fc20260915_common as C
    import fc20260915_data as D
    import data.cache_io as cio
    from core.config import cfg
    from core.presets import apply_preset
    from gameplay.labels import compute_label
    from gameplay.pipeline import interpolate_node_global

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    apply_preset(cfg, "v3.3")
    print(f"CACHE={cio.CACHE_DIR} LABEL_TYPE={cfg.LABEL_TYPE}", flush=True)

    Lyt = D.Layout(False)
    print("load TEST engagements…", flush=True)
    E = D.load_engagements(Lyt, "MAIN", ["TEST"], states=True, counts=True)
    names = list(E["names"])
    pred = np.load(PRED_T, allow_pickle=True)
    pred_key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(
            zip(pred["match"].astype(str).tolist(), pred["s"].astype(np.int64).tolist())
        )
    }
    coh = np.load(
        root / "outputs/cohort_role_training_20260915/cohorts/MAIN_TEST_cohort.npz",
        allow_pickle=False,
    )
    keys_T = set(
        zip(
            coh["match"][coh["cohort"] == 1].astype(str).tolist(),
            coh["s"][coh["cohort"] == 1].astype(np.int64).tolist(),
        )
    )

    em, es = E["match"].astype(str), E["s"].astype(np.int64)
    keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    keep &= np.array([(a, int(b)) in keys_T for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    idx = np.where(keep)[0]

    # --- market_event ---
    print("market_event on TEST T…", flush=True)
    y_svi, y_mkt, q, pt, match_arr = [], [], [], [], []
    n_tie = n_miss = n_fail = n_ok = 0
    by_match: Dict[str, List[int]] = defaultdict(list)
    for i in idx.tolist():
        j = pred_key.get((em[i], int(es[i])))
        if j is not None:
            by_match[em[i]].append((i, j))
    t0 = time.time()
    mids = sorted(by_match.keys())
    for mi, mid in enumerate(mids):
        if mi % 500 == 0:
            print(f"  market {mi}/{len(mids)} ok={n_ok}", flush=True)
        pack = cio.load_match_cache(mid)
        if pack is None:
            n_miss += 1
            continue
        tm = build_team_map(pack)
        for i, j in by_match[mid]:
            s = int(E["s"][i])
            ep = int(E["endpoint_h90"][i])
            L = int(E["L"][i])
            # first/last kill approx: s+B .. L — use L and engagement window
            try:
                y = compute_label(
                    pack,
                    tm,
                    s,
                    engage_ts=s,
                    label_end_ts=ep + 1,
                    first_kill_ts=s + int(cfg.TF2_ENGAGE_PRE_KILL_MS),
                    last_kill_ts=L,
                    interp_node_global=interpolate_node_global,
                    anchor_xy=None,
                )
            except Exception:
                n_fail += 1
                continue
            if y is None:
                n_tie += 1
                continue
            n_ok += 1
            y_mkt.append(float(y))
            y_svi.append(float(pred["y"][j]))
            q.append(float(pred["p_q_base"][j]))
            pt.append(float(pred["p_PT_flex"][j]))
            match_arr.append(str(pred["match"][j]))
    y_svi = np.asarray(y_svi)
    y_mkt = np.asarray(y_mkt)
    q = np.asarray(q)
    pt = np.asarray(pt)
    match_arr = np.asarray(match_arr)
    market_block = dict(
        n_ok=n_ok,
        n_tie_or_drop=n_tie,
        n_miss=n_miss,
        n_fail=n_fail,
        wall_s=float(time.time() - t0),
        label_flip_vs_SVI=float(np.mean(y_svi != y_mkt)) if n_ok else None,
        agree_rate=float(np.mean(y_svi == y_mkt)) if n_ok else None,
        delta_brier_q_minus_pt_on_SVI=brier_binary(y_svi, q, match_arr) - brier_binary(y_svi, pt, match_arr)
        if n_ok
        else None,
        delta_brier_q_minus_pt_on_market=brier_binary(y_mkt, q, match_arr) - brier_binary(y_mkt, pt, match_arr)
        if n_ok
        else None,
        note="Same (engage,s)→endpoint_h90+1 window; v3.3 market_event. Not original CoG corpus numbers.",
    )
    print(f"market ok={n_ok} flip={market_block['label_flip_vs_SVI']}", flush=True)

    # --- R0/R1: nextobj on TRAIN_oof + Q_CAL + Q_SELECT + reuse TEST ---
    print("R0/R1 nextobj + models…", flush=True)
    CLASS_ORDER = [
        "Blue",
        "Red",
        "none",
        "game_ended_before_objective",
        "game_ended_in_window_no_objective",
        "tie_ambiguous",
        "observation_censored",
    ]
    cls_i = {c: i for i, c in enumerate(CLASS_ORDER)}

    def scan_role(role_name: str, lab_path: Path, eng_roles: List[str]) -> Dict[str, np.ndarray]:
        lab = np.load(lab_path, allow_pickle=False)
        E_r = D.load_engagements(Lyt, "MAIN", eng_roles, states=True, counts=True)
        e_key = {
            (a, int(b)): i
            for i, (a, b) in enumerate(
                zip(E_r["match"].astype(str).tolist(), E_r["s"].astype(np.int64).tolist())
            )
        }
        rows = []
        for i, (a, b) in enumerate(zip(lab["match"].astype(str).tolist(), lab["s"].astype(np.int64).tolist())):
            j = e_key.get((a, int(b)))
            if j is None:
                continue
            if lab["Y_SVI"][i] < 0:
                continue
            if "missing_score" in lab.files and lab["missing_score"][i] != 0:
                continue
            if E_r["pre_ok"][j] != 1 or E_r["valid_h90"][j] != 1:
                continue
            rows.append((i, j))
        by_m: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, j in rows:
            by_m[str(lab["match"][i])].append((i, j))
        labels, feats, matches, dVs = [], [], [], []
        n_m = n_f = 0
        for mi, mid in enumerate(sorted(by_m.keys())):
            if mi % 400 == 0:
                print(f"  nextobj {role_name} {mi}/{len(by_m)}", flush=True)
            pack = cio.load_match_cache(mid)
            if pack is None:
                n_m += 1
                continue
            tm = build_team_map(pack)
            gend = game_end_ms(pack)
            for i, j in by_m[mid]:
                try:
                    ep = int(E_r["endpoint_h90"][j])
                    lab_u = next_objective_label(pack["events"], tm, ep, gend)
                    mat = material_row(E_r, j, list(E_r["names"]))
                    p_pre = float(lab["p_pre"][i])
                    dV = float(lab["delta_V"][i])
                    tmin = float(lab["s"][i]) / 60000.0
                    ep_min = ep / 60000.0
                    interval = (ep - int(E_r["q_pre"][j])) / 1000.0
                    x = [
                        p_pre,
                        tmin,
                        ep_min,
                        interval,
                        mat["kill_diff"],
                        mat["objective_net"],
                        mat["epic_net"],
                        mat["structure_net"],
                    ]
                except Exception:
                    n_f += 1
                    continue
                labels.append(lab_u)
                feats.append(x)
                matches.append(mid)
                dVs.append(dV)
        return dict(
            y=np.array(labels, dtype=object),
            X=np.asarray(feats, float),
            match=np.asarray(matches),
            dV=np.asarray(dVs, float),
            n_miss=n_m,
            n_fail=n_f,
        )

    # TRAIN_oof uses fold roles — Map via sub_role? TRAIN_oof_h90 is pooled OOF.
    # Engagements: load MAIN TRAIN folds.
    train = scan_role("TRAIN", LAB / "TRAIN_oof_h90.npz", [f"fold{k}" for k in range(5)])
    qcal = scan_role("Q_CAL", LAB / "Q_CAL_h90.npz", ["Q_CAL"])
    qsel = scan_role("Q_SELECT", LAB / "Q_SELECT_h90.npz", ["Q_SELECT"])
    # TEST: reuse existing nextobj file when possible for speed
    test_lab = np.load(NEXTOBJ_TEST, allow_pickle=True)
    test_E = E
    test_rows = []
    tkey = {
        (a, int(b)): i
        for i, (a, b) in enumerate(
            zip(test_lab["match"].astype(str).tolist(), test_lab["s"].astype(np.int64).tolist())
        )
    }
    lab_test = np.load(LAB / "TEST_h90.npz", allow_pickle=False)
    for i, (a, b) in enumerate(zip(lab_test["match"].astype(str).tolist(), lab_test["s"].astype(np.int64).tolist())):
        j_e = None
        # find engagement
        # use pred join
        jp = pred_key.get((a, int(b)))
        if jp is None:
            continue
        jt = tkey.get((a, int(b)))
        if jt is None:
            continue
        # find eng index
        # rebuild from earlier idx join
        test_rows.append((i, a, int(b), jt, jp))

    # Build TEST features from E via match,s
    e_key_test = {
        (a, int(b)): i
        for i, (a, b) in enumerate(zip(em.tolist(), es.tolist()))
    }
    ty, tX, tm_arr, tdV = [], [], [], []
    for i, a, b, jt, jp in test_rows:
        je = e_key_test.get((a, b))
        if je is None or not keep[je]:
            continue
        try:
            mat = material_row(test_E, je, names)
            p_pre = float(lab_test["p_pre"][i])
            dV = float(lab_test["delta_V"][i])
            tmin = b / 60000.0
            ep = int(test_E["endpoint_h90"][je])
            x = [
                p_pre,
                tmin,
                ep / 60000.0,
                (ep - int(test_E["q_pre"][je])) / 1000.0,
                mat["kill_diff"],
                mat["objective_net"],
                mat["epic_net"],
                mat["structure_net"],
            ]
        except Exception:
            continue
        ty.append(str(test_lab["label"][jt]))
        tX.append(x)
        tm_arr.append(a)
        tdV.append(dV)
    test = dict(y=np.array(ty, dtype=object), X=np.asarray(tX, float), match=np.asarray(tm_arr), dV=np.asarray(tdV, float))

    def encode(yobj):
        return np.array([cls_i.get(str(v), cls_i["observation_censored"]) for v in yobj], dtype=int)

    def fit_eval(name, use_dV: bool):
        # Select C on Q_SELECT match-weighted multiclass Brier
        best = None
        for C_reg in (0.01, 0.1, 1.0, 10.0):
            Xtr = train["X"]
            if use_dV:
                Xtr = np.column_stack([Xtr, train["dV"]])
            ytr = encode(train["y"])
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            clf = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                C=C_reg,
                max_iter=500,
                random_state=7,
            )
            clf.fit(Xtr_s, ytr)
            Xs = qsel["X"]
            if use_dV:
                Xs = np.column_stack([Xs, qsel["dV"]])
            proba = clf.predict_proba(scaler.transform(Xs))
            # align classes
            full = np.zeros((len(qsel["y"]), len(CLASS_ORDER)))
            for ci, c in enumerate(clf.classes_):
                full[:, int(c)] = proba[:, ci]
            br = multiclass_brier(encode(qsel["y"]), full, qsel["match"], len(CLASS_ORDER))
            if best is None or br < best["brier_select"]:
                best = dict(C=C_reg, clf=clf, scaler=scaler, brier_select=br, classes=list(map(int, clf.classes_)))
        # TEST
        Xt = test["X"]
        if use_dV:
            Xt = np.column_stack([Xt, test["dV"]])
        proba = best["clf"].predict_proba(best["scaler"].transform(Xt))
        full = np.zeros((len(test["y"]), len(CLASS_ORDER)))
        for ci, c in enumerate(best["clf"].classes_):
            full[:, int(c)] = proba[:, ci]
        br_te = multiclass_brier(encode(test["y"]), full, test["match"], len(CLASS_ORDER))
        # selected Blue|Red only
        y_e = encode(test["y"])
        mask = np.isin(y_e, [cls_i["Blue"], cls_i["Red"]])
        br_sel = None
        if mask.any():
            br_sel = multiclass_brier(y_e[mask], full[mask][:, :2], test["match"][mask], 2)
        return dict(
            name=name,
            C=best["C"],
            brier_Q_SELECT=best["brier_select"],
            brier_TEST=br_te,
            brier_TEST_BlueRed_selected=br_sel,
            n_train=int(len(train["y"])),
            n_select=int(len(qsel["y"])),
            n_test=int(len(test["y"])),
        )

    r0 = fit_eval("R0", use_dV=False)
    r1 = fit_eval("R1", use_dV=True)

    doc = dict(
        schema="SUPPLEMENTARY_E5_MARKET_R0R1_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        task=".ai/tasks/T023.md",
        market_event=market_block,
        r0_r1=dict(
            classes=CLASS_ORDER,
            train_n=int(len(train["y"])),
            qcal_n=int(len(qcal["y"])),
            qsel_n=int(len(qsel["y"])),
            test_n=int(len(test["y"])),
            R0=r0,
            R1=r1,
            delta_brier_R1_minus_R0_TEST=r1["brier_TEST"] - r0["brier_TEST"],
            reading=(
                "R1 adds ΔV to R0 at endpoint; p_post=p_pre+ΔV reparameterization caveat applies. "
                "Not causal proof that SVI is the true fight value. Selected Blue|Red is a restricted denominator."
            ),
        ),
        E6=dict(status="INCOMPLETE", reason="No pre-registered unseen match list / new scrape for confirmatory eval."),
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Supplementary E5 — market_event + R0/R1 (§8)",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        "",
        "## market_event (same window as SVI)",
        "",
        f"n_ok={market_block['n_ok']}; tie/drop={market_block['n_tie_or_drop']}; "
        f"flip vs SVI={market_block['label_flip_vs_SVI']}; "
        f"ΔBrier q−PT on market={market_block['delta_brier_q_minus_pt_on_market']}.",
        "",
        market_block["note"],
        "",
        "## R0 / R1 nextobj models",
        "",
        f"TRAIN n={doc['r0_r1']['train_n']}; Q_SELECT n={doc['r0_r1']['qsel_n']}; TEST n={doc['r0_r1']['test_n']}.",
        "",
        f"- R0 TEST multiclass Brier={r0['brier_TEST']:.5f} (C={r0['C']})",
        f"- R1 TEST multiclass Brier={r1['brier_TEST']:.5f} (C={r1['C']})",
        f"- R1−R0 = {doc['r0_r1']['delta_brier_R1_minus_R0_TEST']:.5f}",
        f"- Blue|Red selected: R0={r0['brier_TEST_BlueRed_selected']}, R1={r1['brier_TEST_BlueRed_selected']}",
        "",
        doc["r0_r1"]["reading"],
        "",
        "## E6",
        "",
        f"`{doc['E6']['status']}` — {doc['E6']['reason']}",
        "",
    ]
    DOCS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS_MD}", flush=True)
    print(f"R1-R0={doc['r0_r1']['delta_brier_R1_minus_R0_TEST']:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
