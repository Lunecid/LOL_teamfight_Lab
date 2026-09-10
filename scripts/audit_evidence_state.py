"""Leak-free audit: how wrong is the held frame at the next frame, and how much does evidence fix?

For every frame j >= 2 of a match, query tau = minute_ts[j] - 1 ms.  Everything the builder may use
lies at or before frame j-1 (about 60 s stale) plus the events in between; frame j itself is the
truth and is never read by the reconstruction.  This is the worst-case age for a held value, so
the numbers here are upper bounds on the error at a typical engagement query (median age ~32 s).

Three estimates are compared per field: HOLD (frame j-1 as is), EVIDENCE (this module), and for
the continuous accumulators the causal EXTRAPOLATION on its own.  Errors are in raw units
(gold, xp, stat points, map units), and the share of participant-frames with any event in the
gap is reported so 'no change' cases do not dilute the picture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = {"level_norm": 18.0, "xp_norm": 20_000.0, "totalGold_norm": 25_000.0, "curGold_norm": 4_000.0,
       "laneCS_norm": 400.0, "jgCS_norm": 250.0, "cs_attackDamage": 600.0, "cs_abilityPower": 1200.0,
       "cs_armor": 500.0, "cs_magicResist": 400.0, "cs_healthMax": 8500.0, "cs_movementSpeed": 800.0,
       "cs_attackSpeed": 3.0, "alive": 1.0, "hp_pct": 1.0, "has_baron": 1.0, "ult_level_norm": 3.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-matches", type=int, default=300)
    ap.add_argument("--cache-dir", type=Path, default=Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13"))
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/evidence_state/audit.json")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LOL_OUTPUT_ROOT"] = str(a.out.parent / "runtime")
    os.environ["LOL_CFG_PRESET"] = "v3.3"
    os.environ["LOL_CFG_OVERRIDES"] = json.dumps({"CACHE_DIRNAME": str(a.cache_dir.resolve()),
        "FIGHT_INDEX_CACHE_ENABLED": False, "FIGHT_INDEX_NUM_WORKERS": 1, "DUMP_FIGHTS": False})
    from data.cache_io import load_match_cache
    from gameplay.evidence_state import EvidenceStateBuilder, load_tables
    from train.temporal_winprob import stable_int

    protocol = json.loads((a.v3_dir / "protocol.json").read_text(encoding="utf-8"))
    ids = sorted(protocol["splits"]["engagement"], key=lambda m: stable_int("audit:7:" + m))[:a.n_matches]
    tables = {}
    err = {k: defaultdict(list) for k in ("hold", "evidence", "extrapolate")}
    pos = {"hold": [], "evidence": [], "evidence_by_kind": defaultdict(list), "by_kind_age": defaultdict(list)}
    exact = {"level": {"hold": [], "evidence": []}, "alive": {"hold": [], "evidence": []}, "has_baron": {"hold": [], "evidence": []}}
    changed = defaultdict(int)
    n_pf, n_events_gap = 0, 0
    started = time.time()
    for k, mid in enumerate(ids, 1):
        pack = load_match_cache(mid)
        if pack is None:
            continue
        patch = str(pack["meta"]["patch"])
        if patch not in tables:
            tables[patch] = load_tables(patch)
        b = EvidenceStateBuilder(pack, patch=patch, tables=tables[patch])
        ts = b.ts
        event_ts = np.array(sorted(int(e["timestamp"]) for e in pack["events"]))
        for j in range(2, len(ts)):
            tau = int(ts[j]) - 1
            est = b.at(tau)
            for pid in range(1, 11):
                truth = b.frame_values(j, pid)
                held = b.frame_values(j - 1, pid)
                ev = est[pid]
                n_pf += 1
                gap_events = int(((event_ts > ts[j - 1]) & (event_ts <= tau)).sum())
                n_events_gap += gap_events > 0
                for f, den in RAW.items():
                    t, h, e = truth[f] * den, held[f] * den, ev[f] * den
                    err["hold"][f].append(abs(t - h))
                    err["evidence"][f].append(abs(t - e))
                    if abs(t - h) > 1e-9:
                        changed[f] += 1
                # extrapolation alone (no bounty / shop) for the accumulators
                if j - 1 >= 1:
                    dt = (ts[j - 1] - ts[j - 2]) / 1000.0
                    age = (tau - ts[j - 1]) / 1000.0
                    for f in ("totalGold_norm", "xp_norm", "laneCS_norm", "jgCS_norm"):
                        den = RAW[f]
                        prev = b.frame_values(j - 2, pid)[f] * den
                        rate = max(0.0, (held[f] * den - prev) / dt) if dt > 0 else 0.0
                        err["extrapolate"][f].append(abs(truth[f] * den - (held[f] * den + rate * age)))
                exact["level"]["hold"].append(round(held["level_norm"] * 18) == round(truth["level_norm"] * 18))
                exact["level"]["evidence"].append(round(ev["level_norm"] * 18) == round(truth["level_norm"] * 18))
                for f in ("alive", "has_baron"):
                    exact[f]["hold"].append(held[f] == truth[f])
                    exact[f]["evidence"].append(ev[f] == truth[f])
                tx, ty = truth["x_norm"] * 16000, truth["y_norm"] * 16000
                dh = float(np.hypot(tx - held["x_norm"] * 16000, ty - held["y_norm"] * 16000))
                de = float(np.hypot(tx - ev["x_norm"] * 16000, ty - ev["y_norm"] * 16000))
                pos["hold"].append(dh)
                pos["evidence"].append(de)
                pos["evidence_by_kind"][int(ev["pos_evidence_kind"])].append((de, dh))
                pos["by_kind_age"][(int(ev["pos_evidence_kind"]), min(5, int(ev["pos_evidence_age_s"] // 10)))].append((de, dh))
        if k % 50 == 0:
            print(f"[audit] {k}/{len(ids)} matches, {n_pf} participant-frames ({time.time()-started:.0f}s)", flush=True)

    def q(v):
        v = np.asarray(v, dtype=float)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.quantile(v, .9)), "n": int(len(v))}
    kinds = {0: "frame", 1: "victim", 2: "killer", 3: "assist", 4: "objective", 5: "shop", 6: "respawn", 7: "dead"}
    out = {"matches": len(ids), "participant_frames": n_pf, "query_age_s": "~60 (tau = next frame - 1 ms; worst case)",
           "share_with_any_event_in_gap": n_events_gap / max(1, n_pf),
           "fields": {f: {"share_changed_between_frames": changed[f] / max(1, n_pf),
                          "hold_abs_error": q(err["hold"][f]), "evidence_abs_error": q(err["evidence"][f]),
                          **({"extrapolate_only_abs_error": q(err["extrapolate"][f])} if f in err["extrapolate"] else {})}
                      for f in RAW},
           "exact_match_rate": {f: {k: float(np.mean(v)) for k, v in d.items()} for f, d in exact.items()},
           "position_error_u": {"hold": q(pos["hold"]), "evidence": q(pos["evidence"]),
                                "by_evidence_kind": {kinds[k]: {"n": len(v), "evidence_mean": float(np.mean([x for x, _ in v])),
                                                                "hold_mean": float(np.mean([y for _, y in v]))}
                                                     for k, v in sorted(pos["evidence_by_kind"].items())},
                                "by_kind_and_age_bin_10s": {f"{kinds[k]}|{a*10}-{a*10+10}s": {"n": len(v),
                                                                "evidence_mean": float(np.mean([x for x, _ in v])),
                                                                "hold_mean": float(np.mean([y for _, y in v]))}
                                                            for (k, a), v in sorted(pos["by_kind_age"].items())}},
           "elapsed_seconds": round(time.time() - started, 1)}
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"participant_frames": n_pf, "share_with_event": out["share_with_any_event_in_gap"],
                      "position_u": {k: out["position_error_u"][k]["mean"] for k in ("hold", "evidence")},
                      "level_exact": out["exact_match_rate"]["level"], "alive_exact": out["exact_match_rate"]["alive"],
                      "gold_mae": {k: out["fields"]["totalGold_norm"][k + "_abs_error"]["mean"] for k in ("hold", "evidence")},
                      "AD_mae": {k: out["fields"]["cs_attackDamage"][k + "_abs_error"]["mean"] for k in ("hold", "evidence")}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
