"""Tests for gameplay/comparison_gates.py (B6 comparison rows: G1 'evr' gate and G5 tau-state gate).

Synthetic cases pin the rules; the real check reproduces the G1 / G5 decisions (pass, nb, nr, in
call order) of outputs/diag_survival_dbscan_20260925/p4_gate_decomp/rows_full.npz on the first 50
matches of records/baselines/matches_r2_1500.txt (patch 15.14 only).
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.config import cfg
from core.presets import PRESETS
from gameplay import comparison_gates as CG
from gameplay.grid_guard import forbid_grid, grid_forbidden
from gameplay.respawn_rules import death_timer_s

CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
R2_LIST = Path("C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/records/baselines/matches_r2_1500.txt")
ROWS = Path("C:/Users/todtj/문서/LOL_Teamfight/outputs/diag_survival_dbscan_20260925/p4_gate_decomp/rows_full.npz")
GATE_COLS = ("G1_pass", "G1_nb", "G1_nr", "G5_pass", "G5_nb", "G5_nr")
CFG = SimpleNamespace(TF2_VALIDITY_RADIUS=1600.0, COORD_NORM_DIV=16000.0, TF2_GRID_STEP_MS=5000)
TM = {p: (100 if p <= 5 else 200) for p in range(1, 11)}


def _kill(ts, killer, victim, xy, assists=()):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "killerId": killer, "victimId": victim,
            "assistingParticipantIds": list(assists), "position": {"x": xy[0], "y": xy[1]}, "bounty": 300}


def synthetic_pack():
    """11 frames (0..600 s).  From frame 5 (300 s): blue 1, 2, 3, 4 and red 6 near (7000, 7000);
    red 9 kills blue 4 at 302 s there (blue 4 respawns at 312 s); red 7 (far on every frame) kills
    blue 3 at 310 s at (7200, 7200)."""
    mt = np.arange(11, dtype=np.int64) * 60_000
    xy = np.zeros((11, 10, 2), dtype=np.float32)
    xy[:, :5] = (2000.0, 2000.0)
    xy[:, 5:] = (13000.0, 13000.0)
    xy[0, :5] = (400.0, 400.0)                  # spawn points (frame 0)
    xy[0, 5:] = (14400.0, 14400.0)
    near = {1: (7100.0, 7000.0), 2: (7000.0, 7100.0), 3: (7000.0, 6900.0), 4: (7050.0, 7050.0), 6: (6900.0, 7000.0)}
    for p, v in near.items():
        xy[5:, p - 1] = v
    events = [_kill(302_000, 9, 4, (7050, 7050)), _kill(310_000, 7, 3, (7200, 7200))]
    return {"minute_ts": mt, "xy_raw_minute": xy, "events": events,
            "meta": {"patch": "15.14", "team_map": dict(TM), "match_id": "SYN"}}


def test_synthetic_g5_and_g1():
    assert death_timer_s("15.14", 1, 302.0) == 10.0 and death_timer_s("15.14", 1, 310.0) == 10.0
    ctx = CG.gate_context(synthetic_pack(), TM, CFG)
    assert not ctx.is_norm and ctx.div == 1.0 and ctx.validity_radius == 1600.0
    tau, ctr = 318_000, (7000.0, 7000.0)
    pos, own, spawn = CG.tau_state_positions(ctx, tau)
    assert tuple(pos[3]) == (400.0, 400.0) and spawn[3] == 1 and own[3] == 0     # blue 4 respawned at 312 s
    assert tuple(pos[6]) == (7200.0, 7200.0) and own[6] == 1                      # red 7 from its own kill
    assert tuple(pos[8]) == (7050.0, 7050.0) and own[8] == 1                      # red 9 from its own kill
    # G5: blue 1, 2 (3 dead, 4 at spawn); red 6, 7, 9
    assert CG.gate_g5(ctx, tau, ctr) == {"pass": 1, "nb": 2, "nr": 3}
    # G1: the grid has red 7 / red 9 back on their far frames at 315 s -> only red 6
    assert CG.gate_g1(ctx, tau, ctr) == {"pass": 0, "nb": 2, "nr": 1}
    g = CG.gate_decisions(ctx, tau, ctr)
    assert (g["G1_pass"], g["G5_pass"]) == (0, 1)
    # before blue 3's death (309.999 s): blue 1, 2, 3 in; blue 4 still dead; red 6 and 9, not 7 yet
    assert CG.gate_g5(ctx, 309_999, ctr) == {"pass": 1, "nb": 3, "nr": 2}


def test_respawn_segment_is_redrawn_from_spawn():
    ctx = CG.gate_context(synthetic_pack(), TM, CFG)
    dense_ts, xy = CG.respawn_aware_grid(ctx)
    assert CG.respawn_aware_grid(ctx)[1] is xy                                   # cached
    k = int(np.searchsorted(dense_ts, 315_000))
    w = (315_000 - 312_000) / (360_000 - 312_000)                                 # to the next frame (6:00)
    assert xy[k, 3, 0] == pytest.approx(400 + w * (7050 - 400), abs=1e-3)
    assert xy[k, 3, 1] == pytest.approx(400 + w * (7050 - 400), abs=1e-3)
    assert xy[k, 0, 0] == pytest.approx(7100.0)                                   # untouched player


def test_radius_boundary_and_min_per_team():
    ctx = CG.gate_context(synthetic_pack(), TM, CFG)
    # centre exactly R = 1600 east of blue 1 (7100, 7000); blue 2 / 3 are farther than R
    assert CG.gate_g5(ctx, 309_999, (7100.0 + 1600.0, 7000.0))["nb"] == 1         # distance == R counts
    assert CG.gate_g5(ctx, 309_999, (7100.0 + 1601.0, 7000.0))["nb"] == 0
    # G1 grid at 305 s: blue 3 is already drawn towards its 310 s death spot (kill overwrite, future data)
    assert CG.gate_g1(ctx, 309_999, (7100.0 + 1599.0, 7000.0))["nb"] == 2
    ctx.min_per_team = 4
    assert CG.gate_g5(ctx, 309_999, (7000.0, 7000.0))["pass"] == 0                # nb = 3 < 4


def test_norm_coordinates_divide_centre_and_radius():
    pack = synthetic_pack()
    pack["xy_raw_minute"] = pack["xy_raw_minute"] / 16000.0
    pack["meta"]["anchor_is_norm"] = True
    ctx = CG.gate_context(pack, TM, CFG)
    assert ctx.is_norm and ctx.div == 16000.0
    assert CG.gate_g5(ctx, 318_000, (7000.0, 7000.0)) == {"pass": 1, "nb": 2, "nr": 3}


def test_gates_for_records_and_legacy_candidates_need_no_grid():
    pack = synthetic_pack()
    ctx = CG.gate_context(pack, TM, CFG)
    rows = CG.gates_for_records(ctx, [{"tau": 318_000, "centroid_x": 7000.0, "centroid_y": 7000.0},
                                      {"engage_ts": 318_000, "fight_center": (7000.0, 7000.0)}])
    assert rows[0] == rows[1] and rows[0]["G5_pass"] == 1 and rows[0]["tau"] == 318_000
    c = SimpleNamespace(**PRESETS["v3.3"], START_OFFSET_MIN=2, FIGHT_CONTEXT_SEC=30)
    with forbid_grid():
        cands = CG.legacy_gate_candidates(pack, TM, c)
    # kills 302 s and 310 s: gap 8 s <= 13.7 s, both near -> one cluster, tau = 302 - 15 = 287 s
    assert [(x["tau"], x["first_kill_ts"], x["last_kill_ts"]) for x in cands] == [(287_000, 302_000, 310_000)]


# ------------------------------------------------------------------------------------ real 15.14
def _rows():
    if not (CACHE.exists() and R2_LIST.exists() and ROWS.exists()):
        pytest.skip("15.14 cache or p4 gate_decomp rows not available")
    z = np.load(ROWS, allow_pickle=False)
    return z["rows"], [str(c) for c in z["cols"]], [str(m) for m in z["mids"]]


def test_reproduces_gate_decomp_g1_g5_on_50_matches(monkeypatch):
    import data.cache_io as cio
    rows, cols, mids = _rows()
    C = {c: i for i, c in enumerate(cols)}
    for k, v in PRESETS["v3.3"].items():                   # the p3 / p4 detector setting
        monkeypatch.setattr(cfg, k, v, raising=False)
    monkeypatch.setattr(cio, "CACHE_DIR", CACHE)
    sel = R2_LIST.read_text(encoding="utf-8").split()[:50]
    mi_of = {m: i for i, m in enumerate(mids)}
    t0 = time.perf_counter()
    n_rows = n_pass = 0
    with forbid_grid():
        for m in sel:
            pack = cio.load_match_cache(m)
            assert pack is not None and pack["meta"]["patch"] == "15.14"
            ctx = CG.gate_context(pack, None, cfg)
            got = []
            for cand in CG.legacy_gate_candidates(pack, ctx.tm, cfg):
                g = CG.gate_decisions(ctx, cand["tau"], cand["fight_center"])
                got.append((cand["tau"],) + tuple(g[c] for c in GATE_COLS))
            ref = rows[rows[:, C["mi"]] == mi_of[m]]
            exp = [tuple(int(v) for v in r[[C["tau"]] + [C[c] for c in GATE_COLS]]) for r in ref]
            assert got == exp, m
            n_rows += len(exp)
            n_pass += sum(e[1] for e in exp)
        assert grid_forbidden()
    assert n_rows == 1432, n_rows
    print(f"[comparison_gates] 50 matches, {n_rows} gate rows, G1 pass {n_pass}, "
          f"{time.perf_counter() - t0:.1f} s")
