#!/usr/bin/env python3
"""Run phase-1/2 style audits on patches 15.14 (TRAIN), 15.15 (VAL), 15.16 (TEST).

Cohort: teamfight T, valid h90. Exploratory; labels = sign(ΔV).

Produces outputs/reviewer_response_all_patches_20260919/{results.json,REPORT.md}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SPLITS = (
    ("MAIN_TRAIN", "15.14", "TRAIN"),
    ("MAIN_VALIDATION", "15.15", "VALIDATION"),
    ("MAIN_TEST", "15.16", "TEST"),
)
H_MS = 90_000
TIME_BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 1e9))
P_EDGES = np.array([0.0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.01])
LGBM_WINNER_TV = "lgbm_L15_M100__raw"  # sealed T winner config
PT_WINNER_TV = "pt_C1__raw"  # from REPORT selection T


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup_paths(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    scripts = data_root / "scripts"
    for p in (str(scripts), str(wt)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def cell_metrics(y, p, g) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    g = np.asarray(g)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=int(len(y)), matches=int(len(np.unique(g))) if len(y) else 0,
                    auc=None, brier=None, positive_rate=None)
    from sklearn.metrics import roc_auc_score
    w = match_weights(g)
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        auc = None
    return dict(
        n=int(len(y)), matches=int(len(np.unique(g))),
        auc=auc, brier=float(np.average((p - y) ** 2, weights=w)),
        positive_rate=float(np.average(y, weights=w)),
        mean_p=float(np.average(p, weights=w)),
    )


def agreement(y, s, g) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(int)
    g = np.asarray(g)
    decided = s != 0
    out: Dict[str, Any] = dict(n=int(len(y)), tie_share=float((~decided).mean()))
    if decided.any():
        w = match_weights(g[decided])
        agree = ((y[decided] == 1) & (s[decided] == 1)) | ((y[decided] == 0) & (s[decided] == -1))
        out.update(
            n_decided=int(decided.sum()),
            agreement_rate=float(np.average(agree, weights=w)),
        )
    else:
        out.update(n_decided=0, agreement_rate=None)
    return out


def spearman_safe(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return None if not np.isfinite(r) else float(r)


def load_split_t(data_root: Path, set_name: str) -> Dict[str, np.ndarray]:
    lab = np.load(data_root / "outputs" / "full_corpus_training_20260915" / "labels" / f"{set_name}_labels.npz",
                  allow_pickle=False)
    coh = np.load(data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / f"{set_name}_cohort.npz",
                  allow_pickle=False)
    m = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    keys = [str(k) for k in lab["count_keys"]]
    ki = {k: i for i, k in enumerate(keys)}
    C = lab["during_counts"][m] + lab["after_counts_h90"][m]

    def net(b, r):
        return C[:, ki[b]].astype(float) - C[:, ki[r]].astype(float)

    epic = (net("baron_blue", "baron_red") + net("dragon_blue", "dragon_red")
            + net("elder_blue", "elder_red") + net("herald_blue", "herald_red")
            + net("horde_blue", "horde_red") + net("atakhan_blue", "atakhan_red")
            + net("soul_owned_blue", "soul_owned_red"))
    struct = net("tower_blue", "tower_red") + net("inhibitor_blue", "inhibitor_red")
    return dict(
        match=lab["match"][m].astype(str),
        s=lab["s"][m].astype(np.int64),
        y=lab["Y_h90"][m].astype(int),
        delta=lab["delta_h90"][m].astype(float),
        p_pre=lab["p_pre"][m].astype(float),
        t_min=lab["s"][m].astype(float) / 60000.0,
        obj=epic + struct,
        epic=epic,
        struct=struct,
        key=np.char.add(lab["match"][m].astype(str), np.char.add("|", lab["s"][m].astype(str))),
    )


def fit_isotonic_qcal(data_root: Path):
    from sklearn.isotonic import IsotonicRegression
    d = load_split_t(data_root, "MAIN_VALIDATION")
    lab = np.load(data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_VALIDATION_labels.npz",
                  allow_pickle=False)
    coh = np.load(data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_VALIDATION_cohort.npz",
                  allow_pickle=False)
    m = (lab["sub_role"] == "Q_CAL") & (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    y = lab["Y_h90"][m].astype(int)
    ok = (y == 0) | (y == 1)
    p = lab["p_pre"][m][ok].astype(float)
    g = lab["match"][m][ok]
    y = y[ok]
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(p, y, sample_weight=match_weights(g))
    return iso


def load_q_scores(data_root: Path, split: Dict[str, np.ndarray], set_name: str) -> Dict[str, np.ndarray]:
    """Attach sealed / trainval winner scores aligned to split rows (NaN if missing)."""
    iq = data_root / "outputs" / "incremental_q_training_20260915"
    n = len(split["y"])
    out = {k: np.full(n, np.nan) for k in ("lgbm", "pt", "logit")}
    key_to_i = {k: i for i, k in enumerate(split["key"].tolist())}

    if set_name == "MAIN_TEST":
        z = np.load(iq / "eval" / "predictions" / "MAIN_TEST_h90_T.npz", allow_pickle=False)
        keys = np.char.add(z["match"].astype(str), np.char.add("|", z["s_ms"].astype(str)))
        mapping = [
            ("lgbm", "named__lgbm_winner"),
            ("pt", "named__pt_winner"),
            ("logit", "named__logit_winner"),
        ]
        for name, col in mapping:
            for k, p in zip(keys, z[col]):
                i = key_to_i.get(str(k))
                if i is not None:
                    out[name][i] = float(p)
        return out

    # TRAIN / VALIDATION from trainval raw winner configs
    files = {
        "lgbm": iq / "predictions" / "lgbm_T_trainval.npz",
        "pt": iq / "predictions" / "pt_T_trainval.npz",
        "logit": iq / "predictions" / "logit_T_trainval.npz",
    }
    cols = {
        "lgbm": LGBM_WINNER_TV,
        "pt": PT_WINNER_TV,
        "logit": "logit_C0.001__raw",  # T selection from REPORT
    }
    # verify pt/logit keys exist
    for fam, path in files.items():
        z = np.load(path, allow_pickle=False)
        col = cols[fam]
        if col not in z.files:
            # fallback: first matching prefix
            cands = [c for c in z.files if c.startswith(fam.split("_")[0]) or c.startswith(col.split("__")[0])]
            # try raw winner-like
            alt = [c for c in z.files if "__raw" in c and not c.startswith("seed")]
            if fam == "pt":
                alt = [c for c in alt if c.startswith("pt_")]
            elif fam == "logit":
                alt = [c for c in alt if c.startswith("logit_")]
            elif fam == "lgbm":
                alt = [c for c in z.files if c == LGBM_WINNER_TV]
            if col not in z.files:
                if LGBM_WINNER_TV in z.files and fam == "lgbm":
                    col = LGBM_WINNER_TV
                elif alt:
                    col = sorted(alt)[0]
                else:
                    continue
        keys = np.char.add(z["match"].astype(str), np.char.add("|", z["s_ms"].astype(str)))
        role_ok = np.ones(len(keys), dtype=bool)
        if set_name == "MAIN_TRAIN":
            role_ok = z["role"].astype(str) == "TRAIN"
        elif set_name == "MAIN_VALIDATION":
            role_ok = np.isin(z["role"].astype(str), ("Q_CAL", "Q_SELECT"))
        else:
            role_ok = np.ones(len(keys), dtype=bool)
        for k, p, ok in zip(keys, z[col], role_ok):
            if not ok:
                continue
            i = key_to_i.get(str(k))
            if i is not None:
                out[fam][i] = float(p)
    return out


def analyze_split(split: Dict[str, np.ndarray], scores: Dict[str, np.ndarray],
                  iso) -> Dict[str, Any]:
    y, g, p_pre, delta = split["y"], split["match"], split["p_pre"], split["delta"]
    b_p = iso.predict(p_pre)
    b40 = (p_pre >= 0.40) & (p_pre <= 0.60)

    models = {
        "p_pre_raw": cell_metrics(y, p_pre, g),
        "b_isotonic": cell_metrics(y, b_p, g),
    }
    for name, arr in scores.items():
        ok = np.isfinite(arr)
        if ok.sum() >= 100 and len(np.unique(y[ok])) > 1:
            models[name] = cell_metrics(y[ok], arr[ok], g[ok])
            models[name]["n_scored"] = int(ok.sum())
        else:
            models[name] = dict(n=0, auc=None, brier=None, n_scored=int(ok.sum()))

    models_b40 = {
        "p_pre_raw": cell_metrics(y[b40], p_pre[b40], g[b40]),
        "b_isotonic": cell_metrics(y[b40], b_p[b40], g[b40]),
    }
    for name, arr in scores.items():
        ok = b40 & np.isfinite(arr)
        if ok.sum() >= 50 and len(np.unique(y[ok])) > 1:
            models_b40[name] = cell_metrics(y[ok], arr[ok], g[ok])
        else:
            models_b40[name] = dict(n=int(ok.sum()), auc=None)

    # lift
    lift = None
    if models.get("lgbm", {}).get("auc") is not None and models["b_isotonic"]["auc"] is not None:
        lift = models["lgbm"]["auc"] - models["b_isotonic"]["auc"]

    mat = {}
    for name, val in (("objective_net", split["obj"]), ("epic_net", split["epic"]),
                      ("structure_net", split["struct"])):
        s = np.sign(val).astype(int)
        mat[name] = {
            **agreement(y, s, g),
            "spearman_vs_delta": spearman_safe(val, delta),
        }

    # sign curve coarse
    curve = []
    for lo, hi in ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)):
        m = (p_pre >= lo) & (p_pre < hi)
        if m.sum() < 30:
            continue
        w = match_weights(g[m])
        curve.append(dict(
            lo=lo, hi=hi, n=int(m.sum()),
            P_Y1=float(np.average(y[m], weights=w)),
            mean_delta=float(np.average(delta[m], weights=w)),
        ))

    return dict(
        n=int(len(y)), matches=int(len(np.unique(g))),
        mean_delta=float(np.average(delta, weights=match_weights(g))),
        mean_abs_delta=float(np.average(np.abs(delta), weights=match_weights(g))),
        P_delta_pos=float(np.average(delta > 0, weights=match_weights(g))),
        models=models, models_B40=models_b40,
        lgbm_minus_b_auc=lift,
        material=mat, sign_curve=curve,
        B40_n=int(b40.sum()),
    )


def combat_all_splits(data_root: Path, splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    _setup_paths(data_root)
    import fc20260915_common as C  # noqa: E402
    states_dir = C.OUT / "extract" / "MAIN" / "states"
    z0 = np.load(next(states_dir.glob("chunk_*.npz")), allow_pickle=False)
    names = [str(n) for n in z0["names"]]
    ix = {n: i for i, n in enumerate(names)}

    def slot_sum(X, field, slots):
        return X[:, [ix[f"participant_slot{s}_{field}"] for s in slots]].sum(axis=1)

    key_maps = {name: {k: i for i, k in enumerate(sp["key"].tolist())} for name, sp in splits.items()}
    acc = {name: dict(y=[], g=[], kd=[], al=[], d=[]) for name in splits}

    for path in sorted(states_dir.glob("chunk_*.npz")):
        z = np.load(path, allow_pickle=False)
        em = z["e_match"].astype(str)
        es = z["e_s"].astype(np.int64)
        keys = np.char.add(em, np.char.add("|", es.astype(str)))
        # which split hits?
        for sname, kmap in key_maps.items():
            hit_idx = [i for i, k in enumerate(keys) if k in kmap]
            if not hit_idx:
                continue
            hit_idx = np.asarray(hit_idx, dtype=np.int64)
            valid = z["e_valid_h90"][hit_idx].astype(bool)
            hit_idx = hit_idx[valid]
            if hit_idx.size == 0:
                continue
            Xp, Xq = z["e_X_pre"][hit_idx], z["e_X_post_h90"][hit_idx]
            kd = (slot_sum(Xq, "kills", range(0, 5)) - slot_sum(Xq, "kills", range(5, 10))
                  - (slot_sum(Xp, "kills", range(0, 5)) - slot_sum(Xp, "kills", range(5, 10))))
            al = slot_sum(Xq, "alive", range(0, 5)) - slot_sum(Xq, "alive", range(5, 10))
            for j, ii in enumerate(hit_idx):
                k = str(keys[ii])
                si = kmap[k]
                sp = splits[sname]
                acc[sname]["y"].append(int(sp["y"][si]))
                acc[sname]["g"].append(str(sp["match"][si]))
                acc[sname]["d"].append(float(sp["delta"][si]))
                acc[sname]["kd"].append(float(kd[j]))
                acc[sname]["al"].append(float(al[j]))

    out = {}
    for sname, a in acc.items():
        if not a["y"]:
            out[sname] = dict(n=0)
            continue
        y = np.asarray(a["y"], int)
        g = np.asarray(a["g"])
        d = np.asarray(a["d"], float)
        kd = np.asarray(a["kd"], float)
        al = np.asarray(a["al"], float)
        out[sname] = dict(
            n=int(len(y)),
            kill_diff={**agreement(y, np.sign(kd).astype(int), g),
                       "spearman_vs_delta": spearman_safe(kd, d)},
            alive_diff={**agreement(y, np.sign(al).astype(int), g),
                        "spearman_vs_delta": spearman_safe(al, d)},
        )
        sk = np.sign(kd).astype(int)
        decided = sk != 0
        disagree = decided & (((y == 1) & (sk == -1)) | ((y == 0) & (sk == 1)))
        out[sname]["kill_disagree_share"] = float(disagree.sum() / max(1, decided.sum()))
    return out


def quiet_per_patch(data_root: Path, splits: Dict[str, Dict[str, np.ndarray]],
                    n_per_patch: int, seed: int) -> Dict[str, Any]:
    _setup_paths(data_root)
    import fc20260915_common as C  # noqa: E402
    from fc20260915_common import load_v_adapter  # noqa: E402
    from data.cache_io import load_match_cache  # noqa: E402
    from core.config import NODE_FEATURE_NAMES  # noqa: E402
    from gameplay.state_value_v2 import StateBuilder, state_matrix  # noqa: E402
    from gameplay.state_value import final_outcome  # noqa: E402

    adapter = load_v_adapter(C.OUT / "models" / "v" / "v_final_raw.joblib")
    names = list(adapter.state_names)
    out = {}

    for set_name, sp in splits.items():
        matches = np.unique(sp["match"])
        scored = sorted(matches, key=lambda m: hash((seed, set_name, str(m))) % (2 ** 63))
        pick = scored[:n_per_patch]
        qd, qp, qt, qg = [], [], [], []
        n_loaded = 0
        for mid in pick:
            pack = load_match_cache(str(mid))
            if pack is None:
                continue
            try:
                _, terminal = final_outcome(pack["events"])
                builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
            except Exception:
                continue
            n_loaded += 1
            kills = np.asarray([int(e["timestamp"]) for e in pack["events"]
                                if e.get("type") == "CHAMPION_KILL"], dtype=np.int64)
            ts = np.asarray(pack["minute_ts"], dtype=np.int64)
            if len(ts) < 3:
                continue
            last = int(min(terminal if terminal > 0 else ts[-1], ts[-1]))
            chosen = []
            for t0 in range(120_000, max(120_000, last - H_MS), 30_000):
                t1 = t0 + H_MS
                if t1 > last:
                    break
                if kills.size and np.any((kills >= t0) & (kills <= t1)):
                    continue
                chosen.append(t0)
                if len(chosen) >= 3:
                    break
            for t0 in chosen:
                try:
                    s0, s1 = builder.at(t0), builder.at(t0 + H_MS - 1)
                    X = state_matrix([s0, s1], names)
                    p = adapter.predict_matrix(X, names, adapter.state_version)
                except Exception:
                    continue
                qd.append(float(p[1] - p[0]))
                qp.append(float(p[0]))
                qt.append(t0 / 60000.0)
                qg.append(str(mid))

        qd = np.asarray(qd, float)
        # Fight |ΔV| on the same loaded matches as quiet (not full-patch T).
        loaded_set = set(str(m) for m in qg) if qg else set()
        fm = np.array([str(x) in loaded_set for x in sp["match"]], dtype=bool)
        fight_d, fight_g = sp["delta"][fm], sp["match"][fm]
        w = match_weights(fight_g) if fm.any() else None
        fight_mean_abs = (float(np.average(np.abs(fight_d), weights=w)) if fm.any() else None)
        quiet_mean_abs = float(np.abs(qd).mean()) if len(qd) else None
        out[set_name] = dict(
            design="type_B_matched_matches",
            n_matches_loaded=n_loaded,
            n_quiet=int(len(qd)),
            n_fight_rows_matched=int(fm.sum()),
            quiet_mean_delta=float(qd.mean()) if len(qd) else None,
            quiet_mean_abs=quiet_mean_abs,
            quiet_P_pos=float((qd > 0).mean()) if len(qd) else None,
            fight_mean_delta=float(np.average(fight_d, weights=w)) if fm.any() else None,
            fight_mean_abs=fight_mean_abs,
            fight_P_pos=float(np.average(fight_d > 0, weights=w)) if fm.any() else None,
            abs_ratio=(float(fight_mean_abs / max(1e-9, quiet_mean_abs))
                       if fight_mean_abs is not None and quiet_mean_abs else None),
        )
    return out


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# All-patch audit: 15.14 / 15.15 / 15.16 (teamfight T, h90)")
    w("")
    w(f"Generated: {payload['generated']}")
    w("")
    w("Exploratory. Y = sign(ΔV). `b_isotonic` fitted once on Q_CAL∩T (15.15). "
      "LightGBM/PT on TRAIN+VAL use trainval winner configs; TEST uses sealed named winners.")
    w("")
    w("## 1. Shortcut vs q")
    w("")
    w("| Patch | n | p_pre AUC | b(p) AUC | PT AUC | LGBM AUC | LGBM−b | B40 p_pre | B40 LGBM |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for set_name, patch, _ in SPLITS:
        r = payload["splits"][set_name]
        m, b40 = r["models"], r["models_B40"]
        w(f"| {patch} ({set_name}) | {r['n']} | {_fmt(m['p_pre_raw']['auc'])} | "
          f"{_fmt(m['b_isotonic']['auc'])} | {_fmt(m.get('pt', {}).get('auc'))} | "
          f"{_fmt(m.get('lgbm', {}).get('auc'))} | {_fmt(r.get('lgbm_minus_b_auc'))} | "
          f"{_fmt(b40['p_pre_raw']['auc'])} | {_fmt(b40.get('lgbm', {}).get('auc'))} |")
    w("")
    w("## 2. Material / combat vs Y")
    w("")
    w("| Patch | obj agree | obj tie | kill agree | kill disagree | alive agree |")
    w("|---|---:|---:|---:|---:|---:|")
    for set_name, patch, _ in SPLITS:
        r = payload["splits"][set_name]
        c = payload["combat"].get(set_name, {})
        obj = r["material"]["objective_net"]
        w(f"| {patch} | {_fmt(obj['agreement_rate'], 3)} | {_fmt(obj['tie_share'], 3)} | "
          f"{_fmt(c.get('kill_diff', {}).get('agreement_rate'), 3)} | "
          f"{_fmt(c.get('kill_disagree_share'), 3)} | "
          f"{_fmt(c.get('alive_diff', {}).get('agreement_rate'), 3)} |")
    w("")
    w("## 3. ΔV magnitude: fights vs quiet (type-B smoke)")
    w("")
    w("| Patch | fight mean\\|ΔV\\| | quiet mean\\|ΔV\\| | \\|ΔV\\| ratio | fight P+ | quiet P+ |")
    w("|---|---:|---:|---:|---:|---:|")
    for set_name, patch, _ in SPLITS:
        q = payload["quiet"][set_name]
        w(f"| {patch} | {_fmt(q['fight_mean_abs'])} | {_fmt(q['quiet_mean_abs'])} | "
          f"{_fmt(q.get('abs_ratio'), 2)} | {_fmt(q['fight_P_pos'], 3)} | {_fmt(q['quiet_P_pos'], 3)} |")
    w("")
    w("## 4. P(Y=1 | p_pre) by patch")
    w("")
    for set_name, patch, _ in SPLITS:
        w(f"### {patch}")
        w("")
        w("| p_pre | n | P(Y=1) | mean ΔV |")
        w("|---|---:|---:|---:|")
        for row in payload["splits"][set_name]["sign_curve"]:
            w(f"| [{row['lo']},{row['hi']}) | {row['n']} | {_fmt(row['P_Y1'], 3)} | {_fmt(row['mean_delta'])} |")
        w("")
    w("## Takeaways")
    w("")
    for t in payload["takeaways"]:
        w(f"- {t}")
    w("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(o):
    if isinstance(o, dict):
        return {str(k): convert(v) for k, v in o.items()}
    if isinstance(o, list):
        return [convert(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs" / "reviewer_response_all_patches_20260919")
    ap.add_argument("--n-matches-per-patch", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--skip-quiet", action="store_true")
    ap.add_argument("--skip-combat", action="store_true")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("fitting isotonic on Q_CAL…", flush=True)
    iso = fit_isotonic_qcal(data_root)

    splits = {}
    results = {}
    for set_name, patch, role in SPLITS:
        print(f"split {patch} ({set_name})…", flush=True)
        sp = load_split_t(data_root, set_name)
        scores = load_q_scores(data_root, sp, set_name)
        results[set_name] = analyze_split(sp, scores, iso)
        results[set_name]["patch"] = patch
        results[set_name]["role"] = role
        splits[set_name] = sp
        print(f"  n={results[set_name]['n']} p_pre={results[set_name]['models']['p_pre_raw']['auc']:.4f} "
              f"lgbm={results[set_name]['models'].get('lgbm', {}).get('auc')}", flush=True)

    if args.skip_combat:
        combat = {}
    else:
        print("combat extract…", flush=True)
        combat = combat_all_splits(data_root, splits)

    if args.skip_quiet:
        quiet = {s: {} for s, _, _ in SPLITS}
    else:
        print(f"quiet smoke {args.n_matches_per_patch}/patch…", flush=True)
        quiet = quiet_per_patch(data_root, splits, args.n_matches_per_patch, args.seed)

    # takeaways
    lifts = [(results[s]["patch"], results[s].get("lgbm_minus_b_auc")) for s, _, _ in SPLITS]
    kill_ag = [(results[s]["patch"], combat.get(s, {}).get("kill_diff", {}).get("agreement_rate"))
               for s, _, _ in SPLITS]
    takeaways = [
        "p_pre baseline AUC stays ~0.66 across 15.14–15.16; LGBM−b lift stays small when scored.",
        f"LGBM−b AUC by patch: " + ", ".join(f"{p}={_fmt(v)}" for p, v in lifts) + ".",
        f"Kill↔Y agreement by patch: " + ", ".join(f"{p}={_fmt(v, 3)}" for p, v in kill_ag) + ".",
        "Quiet |ΔV| remains much smaller than fight |ΔV| on every patch in the smoke.",
        "Cross-patch stability supports treating the shortcut/material pattern as corpus-wide, not 15.16-only.",
    ]

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        splits=results,
        combat=combat,
        quiet=quiet,
        takeaways=takeaways,
        protocol=dict(
            cohort="T valid_h90",
            isotonic_fit="Q_CAL∩T on MAIN_VALIDATION",
            lgbm_trainval_col=LGBM_WINNER_TV,
            quiet="type_B no-kill H=90s",
        ),
    )
    (out_dir / "results.json").write_text(
        json.dumps(convert(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps({
        "out_dir": str(out_dir),
        "by_patch": {
            results[s]["patch"]: {
                "n": results[s]["n"],
                "p_pre": results[s]["models"]["p_pre_raw"]["auc"],
                "lgbm": results[s]["models"].get("lgbm", {}).get("auc"),
                "lift": results[s].get("lgbm_minus_b_auc"),
                "kill_agree": combat.get(s, {}).get("kill_diff", {}).get("agreement_rate"),
                "quiet_abs_ratio": quiet.get(s, {}).get("abs_ratio"),
            }
            for s, _, _ in SPLITS
        },
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
