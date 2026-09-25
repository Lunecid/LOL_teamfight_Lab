#!/usr/bin/env python3
"""E1 supplement (descriptive only, does not change the E1 estimate).

Why the E1 argmin sits at the upper G edge: participant-sharing rate of the E1 pair universe
(consecutive kills, same match, positions present, dt >= 0) by time gap and by distance, for
15.14 and 15.15. The estimate in e1_results.json is untouched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import ev4_e1_boundaries as E1  # noqa: E402

DT_EDGES_S = [0, 2, 5, 10, 15, 20, 25, 30, 45, 60, 120, 300, 1e9]
DD_SPLIT = 5500.0


def main() -> int:
    data = E1.load_patches()
    res = {"script": str(HERE), "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
           "dt_edges_s": DT_EDGES_S, "distance_split": DD_SPLIT}
    for p in E1.PATCHES:
        P = E1.prepare(data[p])
        U = P["U"]
        dt = P["dt"][U] / 1000.0
        s = P["s"][U]
        dd = np.hypot(P["x"][1:] - P["x"][:-1], P["y"][1:] - P["y"][:-1])[U]
        b = np.digitize(dt, DT_EDGES_S[1:-1], right=True)  # bins (lo, hi]; first bin includes 0
        rows = []
        for k in range(len(DT_EDGES_S) - 1):
            m = b == k
            near, far = m & (dd <= DD_SPLIT), m & (dd > DD_SPLIT)
            rows.append({"dt_s": [DT_EDGES_S[k], DT_EDGES_S[k + 1]], "n": int(m.sum()),
                         "share_rate": float(s[m].mean()) if m.any() else None,
                         "n_within_D": int(near.sum()), "share_rate_within_D": float(s[near].mean()) if near.any() else None,
                         "n_beyond_D": int(far.sum()), "share_rate_beyond_D": float(s[far].mean()) if far.any() else None})
        res[p] = {"n_pairs_U": int(U.sum()), "overall_share_rate": float(s.mean()), "by_dt": rows}
        print(p, json.dumps(rows), flush=True)
    (E1.OUT / "e1_sharing_by_dt.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
