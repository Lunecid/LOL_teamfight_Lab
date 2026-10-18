#!/usr/bin/env python3
"""E2 §5.2 — mechanical S_hold probe with CACHE_MAIN match packs.

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §5.2
Task: .ai/tasks/T022.md

S_hold: events/frames with timestamp ≤ q_pre; query = endpoint_h90;
snapshot = last held frame. Frozen fit85 V. Not a no-fight potential outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WAVE4 = REPO / "outputs/v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
PRED_T = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E2_S_HOLD_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E2_S_HOLD_20260921.md"
E2_FRAME_JSON = REPO / "docs/SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json"
E2_FRAME_MD = REPO / "docs/SUPPLEMENTARY_E2_FRAME_STRATA_20260921.md"
ARITH_ATOL = 1e-6
SUBSAMPLE_MATCHES = 2000


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


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cohort_keys_T(root: Path) -> set:
    coh = np.load(
        root / "outputs/cohort_role_training_20260915/cohorts/MAIN_TEST_cohort.npz",
        allow_pickle=False,
    )
    m = coh["cohort"] == 1
    return set(zip(coh["match"][m].astype(str).tolist(), coh["s"][m].astype(np.int64).tolist()))


def truncate_pack_to_pre(pack: dict, q_pre: int) -> dict:
    q = int(q_pre)
    mts = np.asarray(pack["minute_ts"], dtype=np.int64)
    keep = mts <= q
    if not keep.any():
        raise ValueError("no frames ≤ q_pre")
    out = dict(pack)
    out["events"] = [e for e in pack["events"] if int(e.get("timestamp", -1)) <= q]
    for k in ("minute_ts", "node_minute", "global_minute", "gold_team_minute", "xy_raw_minute"):
        if k in pack:
            arr = np.asarray(pack[k])
            if arr.shape[0] == len(mts):
                out[k] = arr[keep]
    if "events_ts" in pack:
        ets = np.asarray(pack["events_ts"], dtype=np.int64)
        out["events_ts"] = ets[ets <= q]
    return out


def legacy_at_hold(builder, query_ms: int):
    """Copy of gameplay.state_value.StateBuilder.at with hold-relaxed frame bound."""
    from gameplay.state_value import DRAGONS, OBJECTIVES, SNAPSHOT_FIELDS, State

    q = int(query_ms)
    i = int(np.searchsorted(builder.ts, q, side="right") - 1)
    if i < 0:
        raise ValueError("query before first frame")
    # HOLD: allow q past last held frame; keep last snapshot
    if q > int(builder.ts[-1]):
        i = len(builder.ts) - 1
    snapshot_ms = int(builder.ts[i])
    out: Dict[str, float] = {
        "time_minutes": q / 60000.0,
        "time_minutes_sq": (q / 60000.0) ** 2,
        "snapshot_age_s": (q - snapshot_ms) / 1000.0,
    }
    counts = {t: Counter() for t in (100, 200)}
    kills, deaths = {p: [] for p in builder.tm}, {p: [] for p in builder.tm}
    acquisitions = {t: {o: [] for o in OBJECTIVES} for t in (100, 200)}
    souls = {t: set() for t in (100, 200)}
    unknown_team = 0
    for e in builder.events[: bisect_right(builder.event_ts, q)]:
        typ, ts = e["type"], int(e["timestamp"])
        killer = int(e.get("killerId", 0) or 0)
        if typ == "CHAMPION_KILL":
            victim = int(e.get("victimId", 0) or 0)
            if killer in kills:
                kills[killer].append(ts)
                counts[builder.tm[killer]]["kills"] += 1
            if victim in deaths:
                deaths[victim].append(ts)
            continue
        if typ == "ELITE_MONSTER_KILL":
            team = int(e.get("killerTeamId", 0) or builder.tm.get(killer, 0))
            if team not in counts:
                unknown_team += 1
                continue
            monster = e.get("monsterType", "")
            sub = str(e.get("monsterSubType", "")).upper()
            if monster == "DRAGON" and sub != "ELDER_DRAGON":
                kind = sub[:-7] if sub.endswith("_DRAGON") else sub
                kind = kind if kind in DRAGONS else "OTHER"
                counts[team]["dragon_" + kind] += 1
                counts[team]["dragons"] += 1
            else:
                obj = {
                    "BARON_NASHOR": "baron",
                    "RIFTHERALD": "herald",
                    "HORDE": "horde",
                    "ATAKHAN": "atakhan",
                    "DRAGON": "elder",
                }.get(monster)
                if obj:
                    acquisitions[team][obj].append(ts)
                    counts[team][obj] += 1
        elif typ == "DRAGON_SOUL_GIVEN":
            team = int(e.get("teamId", 0) or 0)
            if team in counts:
                soul = str(e.get("dragonSoul", e.get("soulType", "OTHER"))).upper()
                soul = {"INFERNAL": "FIRE", "CLOUD": "AIR", "OCEAN": "WATER", "MOUNTAIN": "EARTH"}.get(soul, soul)
                souls[team].add(soul if soul in DRAGONS else "OTHER")
            else:
                unknown_team += 1
        elif typ in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
            lost = int(e.get("teamId", 0) or 0)
            if lost not in counts:
                unknown_team += 1
                continue
            team = 300 - lost
            if typ == "TURRET_PLATE_DESTROYED":
                counts[team]["plates"] += 1
            elif e.get("buildingType") == "INHIBITOR_BUILDING":
                counts[team]["inhibitor_kills"] += 1
            else:
                tower = str(e.get("towerType", "OTHER"))
                tower = (
                    tower
                    if tower in ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET")
                    else "OTHER"
                )
                counts[team]["tower_" + tower] += 1
    out["unknown_objective_team_count"] = float(unknown_team)
    for pid in sorted(builder.tm, key=builder.slots.get):
        slot = builder.slots[pid]
        prefix = f"slot{slot}_"
        for name in (*SNAPSHOT_FIELDS, "champion_id"):
            out[prefix + name] = float(builder.node[i, pid - 1, builder.idx[name]])
        out[prefix + "kills"] = float(len(kills[pid]))
        out[prefix + "deaths"] = float(len(deaths[pid]))
        out[prefix + "death_since_snapshot"] = float(any(t > snapshot_ms for t in deaths[pid]))
        out[prefix + "death_last_30s"] = float(any(t > q - 30000 for t in deaths[pid]))
        out[prefix + "death_age_minutes"] = (
            min((q - deaths[pid][-1]) / 60000.0, 10.0) if deaths[pid] else 10.0
        )
        for obj in ("baron", "elder"):
            acq = acquisitions[builder.tm[pid]][obj]
            out[prefix + obj + "_death_since_acquisition"] = float(
                bool(acq) and any(t >= acq[-1] for t in deaths[pid])
            )
    for team, prefix in ((100, "blue_"), (200, "red_")):
        names = [
            "kills",
            "dragons",
            "plates",
            "inhibitor_kills",
            *OBJECTIVES,
            *("dragon_" + d for d in DRAGONS),
            *(
                "tower_" + t
                for t in ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET", "OTHER")
            ),
        ]
        for name in names:
            out[prefix + name] = float(counts[team][name])
        for d in DRAGONS:
            out[prefix + "soul_" + d] = float(d in souls[team])
        out[prefix + "soul_event_recorded"] = float(bool(souls[team]))
        for obj in OBJECTIVES:
            history = acquisitions[team][obj]
            age = (q - history[-1]) / 1000.0 if history else float("inf")
            out[prefix + obj + "_ever"] = float(bool(history))
            out[prefix + obj + "_age_minutes"] = min(age / 60.0, 10.0)
            if obj in ("baron", "elder"):
                for seconds in (60, 120, 180, 300):
                    out[f"{prefix}{obj}_acquired_last_{seconds}s"] = float(age < seconds)
    for key, value in list(out.items()):
        if key.startswith(("blue_", "red_")):
            out[key + "_x_time"] = value * q / 1800000.0
    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError("non-finite state")
    return State(out, q, snapshot_ms)


def build_hold_state(pack: dict, node_names: list, q_pre: int, endpoint: int):
    from gameplay.state_value import StateBuilder as LegacyStateBuilder
    from gameplay.state_value_v2 import STATE_VERSION, StateV2, soul_element

    pack_h = truncate_pack_to_pre(pack, q_pre)
    tm = {int(k): int(v) for k, v in pack_h["meta"]["team_map"].items()}
    participant_order = tuple(sorted(tm, key=lambda pid: (tm[pid], pid)))
    sanitized = dict(pack_h)
    sanitized["meta"] = dict(
        pack_h["meta"],
        role_slots={pid: slot for slot, pid in enumerate(participant_order)},
    )
    events, unassigned = [], []
    for source in pack_h["events"]:
        if source.get("type") != "DRAGON_SOUL_GIVEN":
            events.append(source)
            continue
        if int(source.get("teamId", 0) or 0) not in (100, 200):
            unassigned.append(int(source["timestamp"]))
            continue
        events.append(source)
    sanitized["events"] = events
    q = int(endpoint)
    filtered = []
    for source in sanitized["events"]:
        if int(source.get("timestamp", -1)) > q:
            continue
        event = source
        if source.get("type") == "DRAGON_SOUL_GIVEN":
            event = dict(source, dragonSoul=soul_element(source))
        filtered.append(event)
    legacy = LegacyStateBuilder(dict(sanitized, events=filtered), node_names)
    st = legacy_at_hold(legacy, q)
    values = {
        re.sub(r"^slot(\d+)_", r"participant_slot\1_", key): value for key, value in st.values.items()
    }
    return StateV2(
        values,
        st.query_ms,
        st.snapshot_ms,
        STATE_VERSION,
        bisect_right(sorted(unassigned), q),
    )


def summarize(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, float)
    return dict(
        n=int(len(a)),
        mean=float(np.mean(a)) if len(a) else float("nan"),
        mean_abs=float(np.mean(np.abs(a))) if len(a) else float("nan"),
        median_abs=float(np.median(np.abs(a))) if len(a) else float("nan"),
        p_pos=float(np.mean(a > 0)) if len(a) else float("nan"),
        p_neg=float(np.mean(a < 0)) if len(a) else float("nan"),
        p_zero=float(np.mean(a == 0)) if len(a) else float("nan"),
    )


def run_phase(
    phase: str,
    E,
    idx,
    pidx,
    pred,
    names,
    NODE_FEATURE_NAMES,
    cio,
    ev,
    state_matrix,
    StateBuilder,
    predict_calibrated,
) -> Dict[str, Any]:
    em = E["match"].astype(str)
    by_match: Dict[str, List[int]] = defaultdict(list)
    for local, i in enumerate(idx.tolist()):
        by_match[em[i]].append(local)

    A_l, H_l, P_l = [], [], []
    d_clock, d_update, dV = [], [], []
    hold_age, followup, same_frame, y_main = [], [], [], []
    rebuild_pre_err, rebuild_post_err = [], []
    n_miss = n_fail = n_ok = arith_ok = 0
    t0 = time.time()
    match_ids = sorted(by_match.keys())

    for mi, mid in enumerate(match_ids):
        if mi % 200 == 0:
            print(f"  [{phase}] {mi}/{len(match_ids)} ok={n_ok} miss={n_miss} fail={n_fail}", flush=True)
        pack = cio.load_match_cache(mid)
        if pack is None:
            n_miss += 1
            continue
        try:
            builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
        except Exception:
            n_fail += 1
            continue
        for local in by_match[mid]:
            i = idx[local]
            j = pidx[local]
            q_pre = int(E["q_pre"][i])
            ep = int(E["endpoint_h90"][i])
            try:
                s_pre = builder.at(q_pre)
                s_post = builder.at(ep)
                s_hold = build_hold_state(pack, list(NODE_FEATURE_NAMES), q_pre, ep)
                X = state_matrix([s_pre, s_hold, s_post], names)
                p = predict_calibrated(ev, X)
                a, h, p_ = map(float, p)
                rebuild_pre_err.append(float(np.nanmax(np.abs(X[0] - E["X_pre"][i]))))
                rebuild_post_err.append(float(np.nanmax(np.abs(X[2] - E["X_post_h90"][i]))))
            except Exception:
                n_fail += 1
                continue
            n_ok += 1
            A_l.append(a)
            H_l.append(h)
            P_l.append(p_)
            dc, du, dv = h - a, p_ - h, p_ - a
            d_clock.append(dc)
            d_update.append(du)
            dV.append(dv)
            if abs((dc + du) - dv) <= ARITH_ATOL:
                arith_ok += 1
            hold_age.append(float(s_hold.values["snapshot_age_s"]))
            followup.append((ep - q_pre) / 1000.0)
            same_frame.append(int(E["pre_snapshot"][i] == E["post_snapshot_h90"][i]))
            y_main.append(float(pred["y"][j]))

    if n_ok == 0:
        return dict(status="FAIL", phase=phase, reason="no_ok_rows", n_miss=n_miss, n_fail=n_fail)

    dc = np.asarray(d_clock)
    du = np.asarray(d_update)
    dv = np.asarray(dV)
    y = np.asarray(y_main)
    flip = float(np.mean(y != (du > 0).astype(float)))
    result = dict(
        status="OK",
        phase=phase,
        wall_s=float(time.time() - t0),
        n_matches_attempted=len(match_ids),
        n_ok=n_ok,
        n_cache_miss=n_miss,
        n_fail=n_fail,
        arith_identity_rate=float(arith_ok / n_ok),
        arith_atol=ARITH_ATOL,
        rebuild_vs_stored=dict(
            max_pre_err_p99=float(np.percentile(rebuild_pre_err, 99)),
            max_post_err_p99=float(np.percentile(rebuild_post_err, 99)),
            mean_pre_err=float(np.mean(rebuild_pre_err)),
            mean_post_err=float(np.mean(rebuild_post_err)),
        ),
        A=summarize(np.asarray(A_l)),
        H=summarize(np.asarray(H_l)),
        P=summarize(np.asarray(P_l)),
        delta_V=summarize(dv),
        d_clock=summarize(dc),
        d_update=summarize(du),
        Y_vs_d_update_pos_flip=flip,
        opposite_sign_clock_update_rate=float(np.mean(np.sign(dc) * np.sign(du) < 0)),
        mean_hold_snapshot_age_s=float(np.mean(hold_age)),
        mean_followup_s=float(np.mean(followup)),
        same_frame_share=float(np.mean(same_frame)),
        reading=(
            "Arithmetic decomposition under chosen update order only. "
            "S_hold is not a no-fight potential outcome; held frames may be OOD."
        ),
        cache_dir=str(cio.CACHE_DIR),
        bundle_sha16=sha16(BUNDLE),
    )
    print(
        f"  [{phase}] ok={n_ok} arith={result['arith_identity_rate']:.6f} "
        f"E|d_clock|={result['d_clock']['mean_abs']:.5f} E|d_update|={result['d_update']['mean_abs']:.5f} "
        f"flip={flip:.4f} pre_p99={result['rebuild_vs_stored']['max_pre_err_p99']:.3e}",
        flush=True,
    )
    return result


def write_docs(full: Dict[str, Any], subsample_result: Optional[Dict[str, Any]], cache_n: int) -> None:
    doc = dict(
        schema="SUPPLEMENTARY_E2_S_HOLD_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#5.2",
        task=".ai/tasks/T022.md",
        cache=dict(
            CACHE_MAIN=full.get("cache_dir"),
            n_npz=cache_n,
            note="T019 checked empty runtime/cache; packs live on D: CACHE_MAIN.",
        ),
        subsample=subsample_result,
        full=full,
        construction=dict(
            events="timestamp ≤ q_pre",
            frames="minute_ts ≤ q_pre",
            query="endpoint_h90",
            snapshot="last held frame (≤ q_pre)",
            forbidden="mutating only time_minutes on X_pre",
        ),
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Supplementary E2 — S_hold mechanical probe (§5.2)",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"**task:** {doc['task']}  ",
        f"**CACHE_MAIN:** `{doc['cache']['CACHE_MAIN']}` (npz={cache_n})  ",
        f"**bundle_sha16:** `{full.get('bundle_sha16')}`  ",
        "",
        "## Construction",
        "",
        "Events/frames with timestamp ≤ `q_pre`; query = `endpoint_h90`; "
        "snapshot = last held frame. Frozen fit85 V. Not a no-fight potential outcome.",
        "",
    ]
    for label, block in (("Subsample", subsample_result), ("Full TEST T", full)):
        if not block:
            continue
        lines.append(f"## {label} (`{block.get('phase')}`)")
        lines.append("")
        lines.append(
            f"status=`{block['status']}`; n_ok={block.get('n_ok')}; "
            f"miss={block.get('n_cache_miss')}; fail={block.get('n_fail')}; "
            f"arith_identity={block.get('arith_identity_rate')}; wall_s={block.get('wall_s'):.1f}."
        )
        if block.get("status") == "OK":
            lines.append("")
            lines.append("| Quantity | mean | mean_abs | p_pos |")
            lines.append("|---|---:|---:|---:|")
            for key in ("delta_V", "d_clock", "d_update"):
                s = block[key]
                lines.append(f"| {key} | {s['mean']:.5f} | {s['mean_abs']:.5f} | {s['p_pos']:.3f} |")
            lines.append("")
            lines.append(
                f"Y vs 1[d_update>0] flip={block['Y_vs_d_update_pos_flip']:.4f}; "
                f"opposite-sign clock/update={block['opposite_sign_clock_update_rate']:.3f}; "
                f"mean hold snapshot_age_s={block['mean_hold_snapshot_age_s']:.1f}."
            )
            r = block["rebuild_vs_stored"]
            lines.append(
                f"Rebuild vs stored X: pre p99 maxabs={r['max_pre_err_p99']:.3e}, "
                f"post p99={r['max_post_err_p99']:.3e}."
            )
            lines.append("")
            lines.append(block["reading"])
        lines.append("")
    DOCS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DOCS_JSON}", flush=True)
    print(f"wrote {DOCS_MD}", flush=True)


def update_e2_frame_banner(full: Dict[str, Any]) -> None:
    if not E2_FRAME_JSON.is_file():
        return
    doc = json.loads(E2_FRAME_JSON.read_text(encoding="utf-8"))
    doc["section_5_2_S_hold"] = dict(
        status="DONE" if full.get("status") == "OK" else "FAIL",
        report="docs/SUPPLEMENTARY_E2_S_HOLD_20260921.md",
        n_ok=full.get("n_ok"),
        arith_identity_rate=full.get("arith_identity_rate"),
        E_abs_d_clock=full.get("d_clock", {}).get("mean_abs"),
        E_abs_d_update=full.get("d_update", {}).get("mean_abs"),
        cache="CACHE_MAIN on D:",
    )
    E2_FRAME_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if E2_FRAME_MD.is_file():
        md = E2_FRAME_MD.read_text(encoding="utf-8")
        import re as _re

        md2 = _re.sub(
            r"## §5\.2 Mechanical S_hold probe\n\n.*?(?=\n## )",
            "## §5.2 Mechanical S_hold probe\n\n"
            f"**status:** `{doc['section_5_2_S_hold']['status']}` (T022; CACHE_MAIN).  \n\n"
            f"See [`SUPPLEMENTARY_E2_S_HOLD_20260921.md`](SUPPLEMENTARY_E2_S_HOLD_20260921.md): "
            f"n_ok={full.get('n_ok')}, arith={full.get('arith_identity_rate')}, "
            f"E|d_clock|={full.get('d_clock', {}).get('mean_abs')}, "
            f"E|d_update|={full.get('d_update', {}).get('mean_abs')}.\n\n",
            md,
            count=1,
            flags=_re.S,
        )
        E2_FRAME_MD.write_text(md2, encoding="utf-8")


def main() -> int:
    root = data_root()
    setup(root)
    import fc20260915_common as C
    import fc20260915_data as D
    import data.cache_io as cio
    from core.config import NODE_FEATURE_NAMES
    from gameplay.state_value_v2 import StateBuilder, state_matrix
    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    n_cache = sum(1 for _ in cio.CACHE_DIR.glob("*.npz")) if cio.CACHE_DIR.is_dir() else 0
    print(f"CACHE_DIR={cio.CACHE_DIR} npz={n_cache}", flush=True)
    if n_cache == 0:
        raise SystemExit("CACHE_MAIN empty")

    ev = load_evaluator(BUNDLE)
    L = D.Layout(False)
    print("load TEST engagements…", flush=True)
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=False)
    names = list(E["names"])
    keys = cohort_keys_T(root)
    pred = np.load(PRED_T, allow_pickle=True)
    pred_key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(
            zip(pred["match"].astype(str).tolist(), pred["s"].astype(np.int64).tolist())
        )
    }

    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    q_pre_all = E["q_pre"].astype(np.int64)
    keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    key_s = es if in_coh[keep].sum() else q_pre_all
    if in_coh[keep].sum() == 0:
        in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), q_pre_all.tolist())], dtype=bool)
    keep &= in_coh
    idx = np.where(keep)[0]
    rows, pidx = [], []
    for i in idx.tolist():
        j = pred_key.get((em[i], int(key_s[i])))
        if j is not None:
            rows.append(i)
            pidx.append(j)
    idx = np.asarray(rows, int)
    pidx = np.asarray(pidx, int)
    print(f"eligible rows={len(idx)}", flush=True)

    matches = sorted(set(em[idx].tolist()))
    match_rank = sorted(matches, key=lambda m: hashlib.sha256(m.encode()).hexdigest())
    subsample_result = None
    for phase, mset in (("subsample", set(match_rank[:SUBSAMPLE_MATCHES])), ("full", set(matches))):
        print(f"=== phase {phase} n_matches={len(mset)} ===", flush=True)
        mask = np.array([em[i] in mset for i in idx], dtype=bool)
        result = run_phase(
            phase,
            E,
            idx[mask],
            pidx[mask],
            pred,
            names,
            NODE_FEATURE_NAMES,
            cio,
            ev,
            state_matrix,
            StateBuilder,
            predict_calibrated,
        )
        if phase == "subsample":
            subsample_result = result
            if result.get("status") != "OK":
                write_docs(result, None, n_cache)
                return 1
        else:
            write_docs(result, subsample_result, n_cache)
            update_e2_frame_banner(result)
            return 0 if result.get("status") == "OK" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
