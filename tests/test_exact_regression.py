"""Regression reproductions for the v4-exact stage 1 (plan test 3), patch 15.14 only.

(a) The legacy detector (preset v3.3, gameplay.fights.detect_fights, the p3 'ref' arm: frame
    survival, greedy diameter split, no injected helper) on the 15.14 matches of
    records/baselines/matches_r2_1500.txt must give exactly the ref-arm rows of
    outputs/diag_survival_dbscan_20260925/p3/engagements_ref_refdb_eps3927.tsv for those matches.
    The ref arm was written by three p3 runs (tags refdb_eps3927, ev_eps3927, evr_eps3927); on these
    matches the three files are identical (4,394 rows, 1,498 matches with rows).  Only rows whose
    match id is in the 15.14 list are read from the TSV; nothing else is counted.
(b) E1: the 15.14-only distance crossing 4,259.83 (1,918,726 consecutive-kill pairs with
    0 < dt <= 13.7 s) of p2b distance_boundary_full.json, recomputed with the functions of
    scripts/exact_v4/ev4_e1_boundaries.py (prepare, check_c) on the 15.14 kill rows only.

Both are marked slow (run times are printed with -s).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from core.config import cfg
from core.presets import PRESETS

WT = Path(__file__).resolve().parents[1]
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
BASE = Path("C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/records/baselines")
R2_LIST = BASE / "matches_r2_1500.txt"
P3 = Path("C:/Users/todtj/문서/LOL_Teamfight/outputs/diag_survival_dbscan_20260925/p3")
REF_TSV = P3 / "engagements_ref_refdb_eps3927.tsv"
E1_SCRIPT = WT / "scripts" / "exact_v4" / "ev4_e1_boundaries.py"

E1_CROSSING_1514 = 4259.833437826341
E1_N_PAIRS_1514 = 1918726


def _earliest_kill_key(pack, f):
    """p3_detection_arms.earliest_kill_key."""
    fk = int(f.get("first_kill_ts") or -1)
    for e in pack.get("events") or []:
        if e.get("type") == "CHAMPION_KILL" and int(e.get("timestamp", -1) or -1) == fk:
            return (fk, int(e.get("killerId", 0) or 0), int(e.get("victimId", 0) or 0))
    return None


def _keys_of(mid, pack, fights):
    """p3_detection_arms.keys_of, as the TSV row strings."""
    out = []
    for f in fights:
        if not isinstance(f, dict):
            continue
        k = _earliest_kill_key(pack, f)
        if k is None:
            continue
        cb, cr = int(f.get("det_cluster_blue", -1)), int(f.get("det_cluster_red", -1))
        row = (mid, *k, min(cb, cr) if cb >= 0 and cr >= 0 else -1, int(f.get("engage_ts", -1)),
               int(f.get("det_present_blue", -1)), int(f.get("det_present_red", -1)))
        out.append(tuple(str(v) for v in row))
    return out


@pytest.mark.slow
def test_legacy_detector_reproduces_p3_ref_arm_on_1514(monkeypatch, capsys):
    if not (CACHE.exists() and R2_LIST.exists() and REF_TSV.exists()):
        pytest.skip("15.14 cache, match list or p3 ref TSV not available")
    import data.cache_io as cio
    import gameplay.fights as F
    t0 = time.perf_counter()
    for k, v in PRESETS["v3.3"].items():
        monkeypatch.setattr(cfg, k, v, raising=False)
    monkeypatch.setattr(cio, "CACHE_DIR", CACHE)
    # the ref arm runs the detector's own helpers (p3 restores them after every arm)
    for name in ("_validate_teamfight_at_engage", "_split_kill_cluster_spatial", "_build_5s_position_grid"):
        assert getattr(F, name).__module__ == "gameplay.fights", name
    mids = R2_LIST.read_text(encoding="utf-8").split()
    mset = set(mids)
    ref = []
    with open(REF_TSV, encoding="utf-8") as fh:
        header = next(fh).rstrip("\n").split("\t")
        assert header == ["match_id", "first_kill_ts", "killer", "victim", "n_min", "engage_ts",
                          "present_blue", "present_red"]
        for line in fh:
            mid = line.split("\t", 1)[0]
            if mid in mset:                                    # 15.14 list only; nothing else is kept
                ref.append(tuple(line.rstrip("\n").split("\t")))
    got = []
    n_patch = 0
    level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        for mid in mids:
            pack = cio.load_match_cache(mid)
            assert pack is not None, mid
            assert pack["meta"]["patch"] == "15.14", mid         # hard rule: 15.14 only
            n_patch += 1
            tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
            got.extend(_keys_of(mid, pack, F.detect_fights(dict(pack), tm)))
    finally:
        logging.disable(level)
    assert n_patch == len(mids) == 1500
    assert len(ref) == 4394 and len({r[0] for r in ref}) == 1498
    assert sorted(got) == sorted(ref)
    # same order within every match (the TSV is in cache-listing order across matches)
    by = {}
    for r in got:
        by.setdefault(r[0], []).append(r)
    by_ref = {}
    for r in ref:
        by_ref.setdefault(r[0], []).append(r)
    assert by == by_ref
    with capsys.disabled():
        print(f"\n[regression a] {len(mids)} matches, {len(got)} engagements == p3 ref arm, "
              f"{time.perf_counter() - t0:.1f} s")


def _load_e1():
    """Import the E1 script as a module; its sys.path insertions are undone afterwards."""
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("ev4_e1_boundaries_under_test", E1_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


@pytest.mark.slow
def test_e1_1514_crossing_4259_83(capsys):
    if not E1_SCRIPT.exists():
        pytest.skip("ev4_e1_boundaries.py not available")
    E1 = _load_e1()
    if not (E1.KILLS.exists() and E1.MATCHES.exists() and E1.P2B_JSON.exists()):
        pytest.skip("diag p1 kill table or p2b result not available")
    t0 = time.perf_counter()
    # E1.load_patches restricted to 15.14: the patch filter is applied before any other use
    M = json.loads(E1.MATCHES.read_text(encoding="utf-8"))
    keep_m = np.array([m["patch"] for m in M]) == "15.14"
    with np.load(E1.KILLS) as z:
        sel = keep_m[z["match_idx"]]
        K = {c: z[c][sel] for c in ("t_ms", "killer", "victim", "assist_mask", "x", "y", "match_idx")}
    del sel
    P = E1.prepare(K)
    out = E1.check_c(P)
    x = out["extra_15.14_crossing"]
    assert x["n"] == x["ref_n"] == E1_N_PAIRS_1514
    assert x["ref"] == pytest.approx(E1_CROSSING_1514, abs=1e-9)
    assert x["reproduced"] == pytest.approx(E1_CROSSING_1514, abs=1e-6)
    assert x["equal"] is True
    assert out["n_pairs_15.14_same_rule"] == E1_N_PAIRS_1514
    with capsys.disabled():
        print(f"\n[regression b] E1 15.14 crossing {x['reproduced']:.6f} on {x['n']} pairs, "
              f"{time.perf_counter() - t0:.1f} s")
