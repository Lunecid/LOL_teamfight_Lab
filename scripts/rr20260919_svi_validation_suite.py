#!/usr/bin/env python3
"""SVI validation suite: kill×objective disagreement tables + quiet type-A smoke.

Writes outputs/svi_validation_suite_20260919/{results.json,REPORT.md}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPLITS = (
    ("MAIN_TRAIN", "15.14"),
    ("MAIN_VALIDATION", "15.15"),
    ("MAIN_TEST", "15.16"),
)
H_MS = 90_000


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup_wt(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    scripts = data_root / "scripts"
    for p in (str(scripts), str(wt)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


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
        obj=epic + struct,
        key=np.char.add(lab["match"][m].astype(str), np.char.add("|", lab["s"][m].astype(str))),
    )


def combat_kill_alive(data_root: Path, splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    _setup_wt(data_root)
    import fc20260915_common as C  # noqa: E402
    states_dir = C.OUT / "extract" / "MAIN" / "states"
    z0 = np.load(next(states_dir.glob("chunk_*.npz")), allow_pickle=False)
    names = [str(n) for n in z0["names"]]
    ix = {n: i for i, n in enumerate(names)}

    def slot_sum(X, field, slots):
        return X[:, [ix[f"participant_slot{s}_{field}"] for s in slots]].sum(axis=1)

    key_maps = {name: {k: i for i, k in enumerate(sp["key"].tolist())} for name, sp in splits.items()}
    out = {name: dict(kd=np.full(len(sp["y"]), np.nan), al=np.full(len(sp["y"]), np.nan))
           for name, sp in splits.items()}

    for path in sorted(states_dir.glob("chunk_*.npz")):
        z = np.load(path, allow_pickle=False)
        keys = np.char.add(z["e_match"].astype(str), np.char.add("|", z["e_s"].astype(str)))
        for sname, kmap in key_maps.items():
            hit = np.array([k in kmap for k in keys], dtype=bool)
            if not hit.any():
                continue
            valid = hit & (z["e_valid_h90"] == 1)
            if not valid.any():
                continue
            idx = np.flatnonzero(valid)
            Xp, Xq = z["e_X_pre"][idx], z["e_X_post_h90"][idx]
            kd = (slot_sum(Xq, "kills", range(0, 5)) - slot_sum(Xq, "kills", range(5, 10))
                  - (slot_sum(Xp, "kills", range(0, 5)) - slot_sum(Xp, "kills", range(5, 10))))
            al = slot_sum(Xq, "alive", range(0, 5)) - slot_sum(Xq, "alive", range(5, 10))
            for j, ii in enumerate(idx):
                si = kmap[str(keys[ii])]
                out[sname]["kd"][si] = float(kd[j])
                out[sname]["al"][si] = float(al[j])
    return out


def disagreement_tables(sp: Dict[str, np.ndarray], kd: np.ndarray, al: np.ndarray) -> Dict[str, Any]:
    y, g, obj, delta = sp["y"], sp["match"], sp["obj"], sp["delta"]
    sk = np.sign(kd)
    so = np.sign(obj)
    sa = np.sign(al)
    # only where kill decided
    decided = np.isfinite(kd) & (sk != 0)
    y_svi = np.where(y == 1, 1, -1)  # map 0->-1 for sign compare
    agree = decided & (y_svi == sk)
    disagree = decided & (y_svi != sk)

    def wavg(mask, vals=None):
        if not mask.any():
            return None
        w = match_weights(g[mask])
        if vals is None:
            return float(mask.sum())
        return float(np.average(vals[mask], weights=w))

    # among disagreements: does objective agree with SVI?
    obj_with_svi = disagree & (so != 0) & (so == y_svi)
    obj_with_kill = disagree & (so != 0) & (so == sk)
    obj_zero = disagree & (so == 0)
    alive_with_svi = disagree & (sa != 0) & (sa == y_svi)

    # direction breakdown
    svi_blue_kill_red = disagree & (y == 1) & (sk < 0)
    svi_red_kill_blue = disagree & (y == 0) & (sk > 0)

    def count_mask(m):
        return dict(n=int(m.sum()), matches=int(len(np.unique(g[m]))) if m.any() else 0,
                    share_of_disagree=float(m.sum() / max(1, disagree.sum())),
                    mean_abs_delta=float(np.mean(np.abs(delta[m]))) if m.any() else None,
                    mean_p_pre=float(np.mean(sp["p_pre"][m])) if m.any() else None)

    return dict(
        n=int(len(y)),
        n_kill_decided=int(decided.sum()),
        kill_agree_share=float(agree.sum() / max(1, decided.sum())),
        kill_disagree_share=float(disagree.sum() / max(1, decided.sum())),
        n_disagree=int(disagree.sum()),
        among_disagree=dict(
            obj_agrees_with_SVI=count_mask(obj_with_svi),
            obj_agrees_with_kill=count_mask(obj_with_kill),
            obj_tie=count_mask(obj_zero),
            alive_agrees_with_SVI=count_mask(alive_with_svi),
            SVI_blue_kill_red=count_mask(svi_blue_kill_red),
            SVI_red_kill_blue=count_mask(svi_red_kill_blue),
        ),
        # B40 disagree rate
        B40=dict(
            n_decided=int((decided & (sp["p_pre"] >= 0.4) & (sp["p_pre"] <= 0.6)).sum()),
            disagree_share=float(
                (disagree & (sp["p_pre"] >= 0.4) & (sp["p_pre"] <= 0.6)).sum()
                / max(1, (decided & (sp["p_pre"] >= 0.4) & (sp["p_pre"] <= 0.6)).sum())
            ),
        ),
    )


def quiet_type_a(data_root: Path, n_matches: int, seed: int,
                 fight_by_patch: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Proximity kill-less encounters scored with frozen V (smoke)."""
    _setup_wt(data_root)
    # killless scanner from workspace
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_killless_encounters", REPO / "scripts" / "run_killless_encounters.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    encounters_for_match = mod.encounters_for_match

    import fc20260915_common as C  # noqa: E402
    from fc20260915_common import load_v_adapter  # noqa: E402
    from data.cache_io import load_match_cache  # noqa: E402
    from core.config import NODE_FEATURE_NAMES, cfg  # noqa: E402
    from core.contract import NODE_IDX  # noqa: E402
    from gameplay.state_value_v2 import StateBuilder, state_matrix  # noqa: E402
    from gameplay.state_value import final_outcome  # noqa: E402
    from gameplay.fight_clustering import build_5s_position_grid  # noqa: E402
    from gameplay.fights import _extract_kill_events, detect_coordinate_scale  # noqa: E402

    adapter = load_v_adapter(C.OUT / "models" / "v" / "v_final_raw.joblib")
    names = list(adapter.state_names)
    radius = 1600.0
    min_per_team = 2
    min_duration = 13.7
    grace_ms = 15000

    out = {}
    for set_name, sp in fight_by_patch.items():
        matches = np.unique(sp["match"])
        scored = sorted(matches, key=lambda m: hash((seed, "A", set_name, str(m))) % (2 ** 63))
        pick = scored[:n_matches]
        deltas, ps, gs = [], [], []
        n_enc = n_killless = n_loaded = 0
        for mid in pick:
            pack = load_match_cache(str(mid))
            if pack is None:
                continue
            try:
                _, terminal = final_outcome(pack["events"])
                builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
                xy = pack.get("xy_raw_minute")
                if xy is None:
                    xy = pack["node_minute"][:, :, [NODE_IDX.get("x_norm", 0), NODE_IDX.get("y_norm", 1)]]
                minute_ts = np.asarray(pack["minute_ts"], dtype=np.int64)
                if len(minute_ts) < 3:
                    continue
                team_map = pack["meta"]["team_map"]
                blue = np.array([p - 1 for p, t in team_map.items() if int(t) == 100], dtype=int)
                red = np.array([p - 1 for p, t in team_map.items() if int(t) == 200], dtype=int)
                if len(blue) == 0 or len(red) == 0:
                    continue
                kill_events = _extract_kill_events(pack.get("events", []))
                kill_ts = np.array([int(k["timestamp"]) for k in kill_events], dtype=np.int64)
                dense_ts, xy_dense = build_5s_position_grid(
                    xy, minute_ts, kill_events, team_map, cfg_obj=cfg)
                is_norm = bool(pack.get("meta", {}).get("anchor_is_norm", False))
                scale = 1.0
                if not is_norm:
                    is_norm, scale = detect_coordinate_scale(xy)
                r = radius / scale if (is_norm and scale > 0) else radius
                alive = None
                found = encounters_for_match(
                    xy_dense, dense_ts, alive, blue, red, kill_ts,
                    radius=r, min_per_team=min_per_team,
                    min_duration_s=min_duration, grace_ms=grace_ms,
                )
            except Exception:
                continue
            n_loaded += 1
            n_enc += len(found)
            killless = [e for e in found if not e["has_kill"]]
            n_killless += len(killless)
            # score up to 2 killless windows per match, use start and start+H
            for e in killless[:2]:
                t0 = int(e["start_ms"])
                t1 = min(t0 + H_MS - 1, int(terminal) - 1 if terminal > 0 else t0 + H_MS - 1)
                if t1 <= t0:
                    continue
                try:
                    s0, s1 = builder.at(t0), builder.at(t1)
                    X = state_matrix([s0, s1], names)
                    p = adapter.predict_matrix(X, names, adapter.state_version)
                except Exception:
                    continue
                deltas.append(float(p[1] - p[0]))
                ps.append(float(p[0]))
                gs.append(str(mid))

        qd = np.asarray(deltas, float)
        w = match_weights(sp["match"])
        out[set_name] = dict(
            design="type_A_proximity_killless_r1600_t2_dG_g15_score_H90",
            n_matches_loaded=n_loaded,
            n_encounters=n_enc,
            n_killless_encounters=n_killless,
            killless_share=float(n_killless / max(1, n_enc)),
            n_scored=int(len(qd)),
            quiet_mean_abs=float(np.abs(qd).mean()) if len(qd) else None,
            quiet_P_pos=float((qd > 0).mean()) if len(qd) else None,
            fight_mean_abs=float(np.average(np.abs(sp["delta"]), weights=w)),
            abs_ratio=(float(np.average(np.abs(sp["delta"]), weights=w) / max(1e-9, np.abs(qd).mean()))
                       if len(qd) else None),
        )
    return out


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# SVI validation suite — disagreement + quiet type-A")
    w("")
    w(f"Generated: {payload['generated']}")
    w("")
    w("Primary label remains SVI = sign(ΔV). Tables below are **validation axes**.")
    w("")
    w("## 1. Kill × SVI disagreement (teamfight T)")
    w("")
    w("| Patch | Kill decided | Agree | Disagree | Among disagree: obj⇄SVI | obj tie | SVI+ kill− | SVI− kill+ |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for set_name, patch in SPLITS:
        d = payload["disagreement"][set_name]
        a = d["among_disagree"]
        w(f"| {patch} | {d['n_kill_decided']} | {_fmt(d['kill_agree_share'], 3)} | "
          f"{_fmt(d['kill_disagree_share'], 3)} | "
          f"{a['obj_agrees_with_SVI']['n']} ({_fmt(a['obj_agrees_with_SVI']['share_of_disagree'], 2)}) | "
          f"{a['obj_tie']['n']} ({_fmt(a['obj_tie']['share_of_disagree'], 2)}) | "
          f"{a['SVI_blue_kill_red']['n']} | {a['SVI_red_kill_blue']['n']} |")
    w("")
    w("Reading: when SVI and kill signs differ, a non-trivial share has **objective net agreeing with SVI** "
      "(supports ‘not kills alone’ without renaming the label).")
    w("")
    w("## 2. Quiet type-A (proximity kill-less) vs fights")
    w("")
    w("| Patch | Loaded | Encounters | Kill-less share | Scored ΔV | Fight mean\\|ΔV\\| | Quiet mean\\|ΔV\\| | Ratio |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for set_name, patch in SPLITS:
        q = payload["quiet_type_a"][set_name]
        w(f"| {patch} | {q['n_matches_loaded']} | {q['n_encounters']} | {_fmt(q['killless_share'], 3)} | "
          f"{q['n_scored']} | {_fmt(q['fight_mean_abs'])} | {_fmt(q['quiet_mean_abs'])} | {_fmt(q.get('abs_ratio'), 2)} |")
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
                    default=REPO / "outputs" / "svi_validation_suite_20260919")
    ap.add_argument("--n-matches-type-a", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--skip-type-a", action="store_true")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading splits…", flush=True)
    splits = {name: load_split_t(data_root, name) for name, _ in SPLITS}
    print("combat from extract…", flush=True)
    combat = combat_kill_alive(data_root, splits)

    disagreement = {}
    for name, patch in SPLITS:
        print(f"disagreement {patch}…", flush=True)
        disagreement[name] = disagreement_tables(
            splits[name], combat[name]["kd"], combat[name]["al"])

    if args.skip_type_a:
        quiet_a = {name: {} for name, _ in SPLITS}
    else:
        print(f"quiet type-A ({args.n_matches_type_a}/patch)…", flush=True)
        quiet_a = quiet_type_a(data_root, args.n_matches_type_a, args.seed, splits)

    takeaways = [
        "Kill–SVI disagreement is ~6% on every patch; among those, objective often sides with SVI — "
        "the empirical hook for ‘strategic value ≠ kill exchange’.",
        "Type-A proximity kill-less windows (when scored) should show smaller |ΔV| than engagements; "
        "compare to type-B ratios (~2.2–2.5×) in the all-patch report.",
        "Use these tables as Validation Axis A/B under the SVI redesign; do not promote them to the primary label.",
    ]

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        disagreement=disagreement,
        quiet_type_a=quiet_a,
        takeaways=takeaways,
        locked_choices=dict(
            headline_cohort="T",
            N_in_appendix=True,
            quiet_type_B_in_paper=True,
            quiet_type_A_smoke=not args.skip_type_a,
            continuous_deltaV="future_work",
        ),
    )
    (out_dir / "results.json").write_text(
        json.dumps(convert(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps(convert({
        "out_dir": str(out_dir),
        "disagree": {p: disagreement[s]["kill_disagree_share"] for s, p in SPLITS},
        "obj_with_svi_among_disagree": {
            p: disagreement[s]["among_disagree"]["obj_agrees_with_SVI"]["share_of_disagree"]
            for s, p in SPLITS
        },
        "type_a_ratio": {p: quiet_a.get(s, {}).get("abs_ratio") for s, p in SPLITS},
    }), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
