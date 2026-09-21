#!/usr/bin/env python3
"""SVI validation on ONE cohort: the 210k-match main corpus (patches 15.14–15.16).

Cohort contract
---------------
Single study population = designated main corpus of **210,000** Korean matches on
patches **15.14 + 15.15 + 15.16** (MAIN_TRAIN ∪ MAIN_VALIDATION ∪ MAIN_TEST).

Engagement analysis sample = teamfight **T**, `valid_h90`, pooled across those
three patch roles (113,901 T rows).

| Analysis                         | Population                                              |
|----------------------------------|---------------------------------------------------------|
| Material + kill×SVI disagreement | All pooled T rows in the 210k corpus                    |
| Quiet type-B and type-A          | Hash sample of pooled T *matches*; fight |ΔV| = T rows |
|                                  | in *those same loaded matches* only                     |
| p_pre baseline on SVI            | All pooled T rows (honest everywhere)                   |
| Sealed q lift                    | Holdout *role* inside the same 210k corpus (15.16 T);   |
|                                  | cited from phase-1 — not a different match universe     |

Patch (15.14 / 15.15 / 15.16) may appear as a **stratum** of this cohort, never as
a competing headline population.

Writes: outputs/svi_cohort_aligned_20260919/{results.json,REPORT.md}
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
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import rr20260919_svi_validation_suite as suite  # noqa: E402

SPLITS = (
    ("MAIN_TRAIN", "15.14"),
    ("MAIN_VALIDATION", "15.15"),
    ("MAIN_TEST", "15.16"),
)
RAW_MATCHES = 210_000
H_MS = 90_000
POOLED_KEY = "POOLED_210k"


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def sample_match_ids(matches: np.ndarray, n: int, seed: int, tag: str) -> List[str]:
    uniq = np.unique(matches.astype(str))
    scored = sorted(uniq, key=lambda m: hash((seed, tag, str(m))) % (2 ** 63))
    return list(scored[:n])


def pool_splits(data_root: Path) -> Dict[str, np.ndarray]:
    """Concatenate T/valid_h90 rows from 15.14–15.16 into one cohort array dict."""
    parts = []
    for set_name, patch in SPLITS:
        sp = suite.load_split_t(data_root, set_name)
        n = len(sp["y"])
        sp = dict(sp)
        sp["patch"] = np.full(n, patch, dtype=object)
        sp["set_name"] = np.full(n, set_name, dtype=object)
        parts.append(sp)
    out: Dict[str, np.ndarray] = {}
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts])
    return out


def fight_stats_on_matches(sp: Dict[str, np.ndarray], match_ids: Sequence[str]) -> Dict[str, Any]:
    mid = set(str(m) for m in match_ids)
    m = np.array([str(x) in mid for x in sp["match"]], dtype=bool)
    if not m.any():
        return dict(n_rows=0, n_matches=0, mean_abs=None, mean_delta=None, P_pos=None)
    d = sp["delta"][m]
    g = sp["match"][m]
    w = suite.match_weights(g)
    return dict(
        n_rows=int(m.sum()),
        n_matches=int(len(np.unique(g))),
        mean_abs=float(np.average(np.abs(d), weights=w)),
        mean_delta=float(np.average(d, weights=w)),
        P_pos=float(np.average(d > 0, weights=w)),
    )


def quiet_type_b_on_matches(
    data_root: Path, sp: Dict[str, np.ndarray], pick: Sequence[str],
) -> Dict[str, Any]:
    suite._setup_wt(data_root)
    import fc20260915_common as C  # noqa: E402
    from fc20260915_common import load_v_adapter  # noqa: E402
    from data.cache_io import load_match_cache  # noqa: E402
    from core.config import NODE_FEATURE_NAMES  # noqa: E402
    from gameplay.state_value_v2 import StateBuilder, state_matrix  # noqa: E402
    from gameplay.state_value import final_outcome  # noqa: E402

    adapter = load_v_adapter(C.OUT / "models" / "v" / "v_final_raw.joblib")
    names = list(adapter.state_names)
    deltas: List[float] = []
    loaded: List[str] = []
    for mid in pick:
        pack = load_match_cache(str(mid))
        if pack is None:
            continue
        try:
            _, terminal = final_outcome(pack["events"])
            builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
        except Exception:
            continue
        loaded.append(str(mid))
        kills = np.asarray(
            [int(e["timestamp"]) for e in pack.get("events", [])
             if str(e.get("type", e.get("eventType", ""))).upper() == "CHAMPION_KILL"],
            dtype=np.int64,
        )
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
            deltas.append(float(p[1] - p[0]))

    qd = np.asarray(deltas, float)
    fight = fight_stats_on_matches(sp, loaded)
    quiet_mean_abs = float(np.abs(qd).mean()) if len(qd) else None
    return dict(
        design="type_B_no_kill_fixed_H90_matched_matches",
        n_matches_requested=len(pick),
        n_matches_loaded=len(loaded),
        n_scored=int(len(qd)),
        quiet_mean_abs=quiet_mean_abs,
        quiet_P_pos=float((qd > 0).mean()) if len(qd) else None,
        fight=fight,
        abs_ratio=(float(fight["mean_abs"] / max(1e-9, quiet_mean_abs))
                   if quiet_mean_abs and fight["mean_abs"] is not None else None),
        match_ids_loaded=loaded,
    )


def quiet_type_a_on_matches(
    data_root: Path, sp: Dict[str, np.ndarray], pick: Sequence[str],
) -> Dict[str, Any]:
    suite._setup_wt(data_root)
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
    radius, min_per_team, min_duration, grace_ms = 1600.0, 2, 13.7, 15000

    deltas: List[float] = []
    loaded: List[str] = []
    n_enc = n_killless = 0
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
            found = encounters_for_match(
                xy_dense, dense_ts, None, blue, red, kill_ts,
                radius=r, min_per_team=min_per_team,
                min_duration_s=min_duration, grace_ms=grace_ms,
            )
        except Exception:
            continue
        loaded.append(str(mid))
        n_enc += len(found)
        killless = [e for e in found if not e["has_kill"]]
        n_killless += len(killless)
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

    qd = np.asarray(deltas, float)
    fight = fight_stats_on_matches(sp, loaded)
    quiet_mean_abs = float(np.abs(qd).mean()) if len(qd) else None
    return dict(
        design="type_A_proximity_killless_r1600_matched_matches",
        n_matches_requested=len(pick),
        n_matches_loaded=len(loaded),
        n_encounters=n_enc,
        n_killless_encounters=n_killless,
        killless_share=float(n_killless / max(1, n_enc)),
        n_scored=int(len(qd)),
        quiet_mean_abs=quiet_mean_abs,
        quiet_P_pos=float((qd > 0).mean()) if len(qd) else None,
        fight=fight,
        abs_ratio=(float(fight["mean_abs"] / max(1e-9, quiet_mean_abs))
                   if quiet_mean_abs and fight["mean_abs"] is not None else None),
        match_ids_loaded=loaded,
    )


def material_block(sp: Dict[str, np.ndarray], combat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    y, g, obj = sp["y"], sp["match"], sp["obj"]
    kd, al = combat["kd"], combat["al"]

    def axis(raw: np.ndarray, y_arr: np.ndarray, g_arr: np.ndarray) -> Dict[str, Any]:
        s = np.sign(raw)
        decided = np.isfinite(raw) & (s != 0)
        if not decided.any():
            return dict(n_decided=0, tie_share=1.0, agree=None)
        w = suite.match_weights(g_arr[decided])
        agree = ((y_arr[decided] == 1) & (s[decided] == 1)) | ((y_arr[decided] == 0) & (s[decided] == -1))
        return dict(
            n_decided=int(decided.sum()),
            tie_share=float((~decided).mean()),
            agree=float(np.average(agree, weights=w)),
        )

    by_patch = {}
    for _, patch in SPLITS:
        m = sp["patch"].astype(str) == patch
        if not m.any():
            continue
        by_patch[patch] = dict(
            n=int(m.sum()),
            n_matches=int(len(np.unique(sp["match"][m]))),
            kill=axis(kd[m], y[m], g[m]),
            objective=axis(obj[m], y[m], g[m]),
            alive=axis(al[m], y[m], g[m]),
        )

    return dict(
        n=int(len(y)),
        n_matches=int(len(np.unique(g))),
        objective=axis(obj, y, g),
        kill=axis(kd, y, g),
        alive=axis(al, y, g),
        by_patch=by_patch,
    )


def p_pre_baseline(sp: Dict[str, np.ndarray]) -> Dict[str, Any]:
    y = sp["y"].astype(int)
    p = sp["p_pre"].astype(float)
    g = sp["match"]
    w = suite.match_weights(g)
    # match-weighted AUC via bootstrap-free sklearn on rows is fine for descriptive;
    # report unweighted AUC (standard) + prevalence.
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = None
    return dict(
        n=int(len(y)),
        n_matches=int(len(np.unique(g))),
        prevalence=float(np.average(y, weights=w)),
        p_pre_auc=auc,
        mean_abs_delta=float(np.average(np.abs(sp["delta"]), weights=w)),
    )


def load_phase1_holdout(repo: Path) -> Optional[Dict[str, Any]]:
    """Sealed q lift on the 15.16 role *inside* the 210k corpus."""
    p = repo / "outputs" / "reviewer_response_phase1_20260919" / "results.json"
    if not p.is_file():
        return None
    z = json.loads(p.read_text(encoding="utf-8"))
    return {
        "source": str(p),
        "role_inside_210k": "MAIN_TEST / patch 15.16 / T",
        "note": "Holdout evaluation role within the same 210k corpus — not a separate study population.",
        "cohort": z.get("cohort"),
        "n_test": z.get("n_test"),
        "n_matches": z.get("n_matches"),
        "models": z.get("models"),
        "models_B40": z.get("models_B40"),
        "contrasts": z.get("contrasts") or z.get("bootstrap"),
    }


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# SVI cohort-aligned validation (210k matches, patches 15.14–15.16)")
    w("")
    w(f"Generated: {payload['generated']}")
    w("")
    w("## Cohort contract")
    w("")
    w(f"- **Single population:** {RAW_MATCHES:,} raw matches on patches **15.14 + 15.15 + 15.16**.")
    w("- **Engagement sample:** pooled teamfight T ∩ valid h90 across those three patch roles.")
    w("- **Material & disagreement:** all pooled T rows.")
    w("- **Quiet A/B:** one hash sample from pooled T matches; fight `|ΔV|` = T rows in those loaded matches.")
    w("- **p_pre on SVI:** all pooled T rows.")
    w("- **Sealed q lift:** 15.16 holdout *role* inside this same 210k corpus (phase-1); not a second cohort.")
    w("- Patch columns below are **strata of the same cohort**, not competing headlines.")
    w("")
    h = payload["pooled"]
    w(f"Pooled T n = {h['material']['n']} rows / {h['material']['n_matches']} matches "
      f"(raw corpus {RAW_MATCHES:,}).")
    w("")
    w("## 1. p_pre baseline + sealed q (same 210k corpus)")
    w("")
    pp = h["p_pre"]
    w(f"- Pooled T `p_pre` AUC vs SVI: **{_fmt(pp.get('p_pre_auc'))}** "
      f"(n={pp['n']}, prevalence={_fmt(pp.get('prevalence'), 3)}, "
      f"mean `|ΔV|`={_fmt(pp.get('mean_abs_delta'))}).")
    p1 = payload.get("phase1_holdout")
    if p1:
        w(f"- Sealed q lift (15.16 role inside 210k): `{p1['source']}` · n={p1.get('n_test')}. "
          "See that report for AUC/Brier vs `p_pre` / `b(p)` / PT / B40.")
    else:
        w("- _phase-1 results.json not found — run `rr20260919_phase1_shortcut_audit.py`._")
    w("")
    w("## 2. Material concordance (pooled T on 210k)")
    w("")
    m = h["material"]
    w("| Axis | Decided n | Tie share | Agree with SVI |")
    w("|---|---:|---:|---:|")
    for name, key in (("Kill diff", "kill"), ("Objective net", "objective"), ("Alive diff", "alive")):
        a = m[key]
        w(f"| {name} | {a['n_decided']} | {_fmt(a['tie_share'], 3)} | {_fmt(a.get('agree'), 3)} |")
    w("")
    w("### Patch strata (same cohort)")
    w("")
    w("| Patch | T rows | Kill agree | Obj agree | Alive agree |")
    w("|---|---:|---:|---:|---:|")
    for _, patch in SPLITS:
        b = m.get("by_patch", {}).get(patch, {})
        if not b:
            continue
        w(f"| {patch} | {b['n']} | {_fmt(b['kill'].get('agree'), 3)} | "
          f"{_fmt(b['objective'].get('agree'), 3)} | {_fmt(b['alive'].get('agree'), 3)} |")
    w("")
    d = h["disagreement"]
    a = d["among_disagree"]
    w("### Kill × SVI disagreement (pooled)")
    w("")
    w(f"- Kill decided {d['n_kill_decided']}: agree {_fmt(d['kill_agree_share'], 3)}, "
      f"disagree {_fmt(d['kill_disagree_share'], 3)}.")
    w(f"- Among disagree: obj⇄SVI {a['obj_agrees_with_SVI']['n']} "
      f"({_fmt(a['obj_agrees_with_SVI']['share_of_disagree'], 2)}); "
      f"obj tie {a['obj_tie']['n']} ({_fmt(a['obj_tie']['share_of_disagree'], 2)}).")
    w(f"- Direction: SVI+ kill− {a['SVI_blue_kill_red']['n']}; "
      f"SVI− kill+ {a['SVI_red_kill_blue']['n']}.")
    w("")
    w("## 3. Quiet references (matched matches from pooled 210k T)")
    w("")
    w(f"Shared sample request: {payload['n_matches_quiet']} matches · seed={payload['seed']}.")
    w("")
    w("| Design | Loaded | Scored | Fight rows (same matches) | Fight mean\\|ΔV\\| | Quiet mean\\|ΔV\\| | Ratio |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for key, label in (("quiet_type_b", "Type-B no-kill H=90s"),
                       ("quiet_type_a", "Type-A proximity kill-less")):
        q = h[key]
        f = q.get("fight") or {}
        w(f"| {label} | {q.get('n_matches_loaded', 0)} | {q.get('n_scored', 0)} | "
          f"{f.get('n_rows', 0)} | {_fmt(f.get('mean_abs'))} | {_fmt(q.get('quiet_mean_abs'))} | "
          f"{_fmt(q.get('abs_ratio'), 2)} |")
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
                    default=REPO / "outputs" / "svi_cohort_aligned_20260919")
    ap.add_argument("--n-matches-quiet", type=int, default=1500,
                    help="shared quiet A/B match budget drawn from pooled 210k T matches")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--skip-quiet", action="store_true")
    args = ap.parse_args(argv)
    data_root = args.data_root or suite._data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"pooling T across {RAW_MATCHES} matches (15.14-15.16)...", flush=True)
    sp = pool_splits(data_root)
    print(f"  pooled T rows={len(sp['y'])} matches={len(np.unique(sp['match']))}", flush=True)

    print("combat from extract…", flush=True)
    combat_all = suite.combat_kill_alive(data_root, {POOLED_KEY: sp})
    combat = combat_all[POOLED_KEY]

    print("material + disagreement…", flush=True)
    mat = material_block(sp, combat)
    dis = suite.disagreement_tables(sp, combat["kd"], combat["al"])
    pp = p_pre_baseline(sp)

    pick = sample_match_ids(sp["match"], args.n_matches_quiet, args.seed, "quiet_shared_210k")
    if args.skip_quiet:
        qb = dict(design="skipped", n_matches_loaded=0, n_scored=0, fight={}, abs_ratio=None)
        qa = dict(design="skipped", n_matches_loaded=0, n_scored=0, fight={}, abs_ratio=None)
    else:
        print(f"quiet type-B on {len(pick)} pooled matches…", flush=True)
        qb = quiet_type_b_on_matches(data_root, sp, pick)
        print(f"quiet type-A on {len(pick)} pooled matches…", flush=True)
        qa = quiet_type_a_on_matches(data_root, sp, pick)

    pooled = dict(
        material=mat,
        disagreement=dis,
        p_pre=pp,
        quiet_type_b=qb,
        quiet_type_a=qa,
    )
    phase1 = load_phase1_holdout(REPO)
    takeaways = [
        "Headline population is the 210k-match corpus (15.14–15.16), not MAIN_TEST alone.",
        "Quiet fight|ΔV| uses the quiet match subset drawn from that pooled T universe.",
        "Sealed q numbers remain the 15.16 holdout *role* inside the same 210k corpus.",
        "Do not cite older MAIN_TEST-only or mixed full-T-vs-smoke quiet ratios.",
    ]
    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        cohort_contract=dict(
            raw_matches=RAW_MATCHES,
            patches=["15.14", "15.15", "15.16"],
            engagement_sample="pooled T valid_h90",
            quiet="hash sample from pooled T matches; fight comparator matched",
            sealed_q="15.16 role inside same corpus",
        ),
        n_matches_quiet=args.n_matches_quiet,
        seed=args.seed,
        phase1_holdout=phase1,
        pooled=pooled,
        takeaways=takeaways,
    )

    def strip_ids(block):
        for k in ("quiet_type_a", "quiet_type_b"):
            if k in block and isinstance(block[k], dict):
                block[k] = {kk: vv for kk, vv in block[k].items() if kk != "match_ids_loaded"}
        return block

    payload_write = convert(payload)
    payload_write["pooled"] = strip_ids(payload_write["pooled"])
    (out_dir / "results.json").write_text(
        json.dumps(payload_write, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps(convert({
        "out_dir": str(out_dir),
        "pooled_t_rows": mat["n"],
        "pooled_t_matches": mat["n_matches"],
        "p_pre_auc": pp.get("p_pre_auc"),
        "kill_disagree": dis["kill_disagree_share"],
        "type_b_ratio": qb.get("abs_ratio"),
        "type_a_ratio": qa.get("abs_ratio"),
        "quiet_fight_rows": (qb.get("fight") or {}).get("n_rows"),
    }), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
