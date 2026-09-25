"""gameplay.exact_population / gameplay.cohorts_exact (v4-exact stage 1, task B1)."""
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from gameplay import exact_population as EP
from gameplay.cohorts_exact import (cohort_from_counts, cohort_label, cohort_record, credited_sets,
                                    kill_credit, team_sets)
from gameplay.event_survival import death_intervals, event_alive
from gameplay.exact_population import (ExactParams, R2_COMPARE_COLUMNS, detect_engagements_exact,
                                       frame_age_ms, isolation_counts, load_exact_pack, r2_tsv_row)

WT = Path(__file__).resolve().parents[1]
BASE = WT.parents[1] / "outputs" / "reest_exact_v4_20260925" / "records" / "baselines"
CACHE = EP.DEFAULT_CACHE_DIR
TM = {i: 100 if i <= 5 else 200 for i in range(1, 11)}
PR = ExactParams(gap_ms=13700, diameter=4264.0)          # R2 / v3.3 boundaries
P_AT = (5000.0, 5000.0)
Q_AT = (12000.0, 12000.0)                                # > D and > merge radius from P_AT
Z_AT = (3000.0, 12000.0)
T = 600_000


def _k(ts, killer, victim, at, assists=()):
    return {"type": "CHAMPION_KILL", "timestamp": int(ts), "killerId": killer, "victimId": victim,
            "assistingParticipantIds": list(assists), "position": {"x": at[0], "y": at[1]}}


def _pack(events, n_frames=40):
    ev = sorted(events, key=lambda e: e["timestamp"])
    return {"meta": {"patch": "15.14", "match_id": "SYN"}, "events": ev,
            "minute_ts": np.arange(n_frames, dtype=np.int64) * 60000}


def _v4(events, params=PR, **kw):
    return detect_engagements_exact(_pack(events, **kw), TM, params, "v4")


# ------------------------------------------------------------------ cohorts
@pytest.mark.parametrize("nb,nr,label,t5", [
    (5, 5, "T", 1), (4, 4, "T", 0), (4, 5, "T", 0), (3, 3, "S", 0), (2, 2, "S", 0), (2, 3, "S", 0),
    (3, 4, "ASYM", 0), (2, 5, "ASYM", 0), (5, 3, "ASYM", 0), (1, 5, "P", 0), (1, 1, "P", 0),
    (0, 0, "P", 0), (5, 0, "P", 0), (1, 2, "P", 0)])
def test_cohort_boundaries(nb, nr, label, t5):
    assert cohort_label(nb, nr) == label
    rec = cohort_from_counts(nb, nr)
    assert (rec["cohort"], rec["t5"], rec["n_min"], rec["n_max"]) == (label, t5, min(nb, nr), max(nb, nr))


def test_credited_sets_both_formats_and_non_champion_killer():
    raw = [_k(T, 0, 3, P_AT, assists=[7, 8]), _k(T + 1, 6, 1, P_AT, assists=[2, 11, 0])]
    ext = [{"killer_id": 0, "victim_id": 3, "assisting_ids": [7, 8]},
           {"killer_id": 6, "victim_id": 1, "assisting_ids": [2, 11, 0]}]
    for kills in (raw, ext):
        blue, red = credited_sets(kills, TM)
        assert blue == {1, 2, 3} and red == {6, 7, 8}
    assert kill_credit({"killerId": 0, "victimId": 0}) == set()
    rec = cohort_record(raw, TM)
    assert (rec["n_blue"], rec["n_red"], rec["cohort"], rec["blue"], rec["red"]) == (3, 3, "S", [1, 2, 3], [6, 7, 8])


def test_team_sets_follow_team_map():
    tm = {1: 200, 2: 200, 3: 200, 4: 200, 5: 200, 6: 100, 7: 100, 8: 100, 9: 100, 10: 100}
    assert team_sets(tm) == (frozenset({6, 7, 8, 9, 10}), frozenset({1, 2, 3, 4, 5}))


# ------------------------------------------------------------------ params
def test_params_require_g_and_d_and_reject_unknown():
    with pytest.raises(TypeError):
        EP._coerce_params({"gap_ms": 14000})
    with pytest.raises(KeyError):
        EP._coerce_params({"gap_ms": 14000, "diameter": 4300.0, "bogus": 1})
    assert EP._coerce_params({"gap_ms": 14000, "diameter": 4300.0}) == ExactParams(14000, 4300.0)
    with pytest.raises(ValueError):
        detect_engagements_exact(_pack([]), TM, PR, "legacy")


def test_params_from_cfg_v4_preset():
    from core.config import cfg
    from core.presets import apply_preset
    c = copy.copy(cfg)
    apply_preset(c, "v4-exact")
    p = ExactParams.from_cfg(c)
    assert (p.gap_ms, p.diameter) == (int(c.TF2_KILL_CLUSTER_GAP_MS), float(c.CLUSTER_MAX_DIAMETER))
    assert (p.pre_kill_ms, p.horizon_ms, p.context_ms, p.start_offset_ms) == (15000, 35000, 30000, 120000)
    assert (p.max_duration_ms, p.merge_max_gap_ms, p.merge_radius, p.clean_max_age_ms) == (60000, 15000, 2000.0, 10000)


# ------------------------------------------------------------------ clean boundary
@pytest.mark.parametrize("fk,clean,age", [(624_999, 1, 9_999), (625_000, 0, 10_000), (615_000, 1, 0)])
def test_clean_boundary(fk, clean, age):
    recs, _ = _v4([_k(fk, 6, 1, P_AT)])
    assert len(recs) == 1
    assert recs[0]["tau"] == fk - 15000
    assert (recs[0]["frame_age_ms"], recs[0]["clean"]) == (age, clean)
    assert frame_age_ms(np.arange(5) * 60000, 59_999) == 59_999 and frame_age_ms(np.arange(5) * 60000 + 10, 5) is None


def test_clean_filter_off_when_zero():
    recs, _ = _v4([_k(625_000, 6, 1, P_AT)], params=ExactParams(13700, 4264.0, clean_max_age_ms=0))
    assert recs[0]["clean"] == 1


# ------------------------------------------------------------------ isolation boundaries
def _a_kills():
    return [_k(T, 6, 1, P_AT, assists=[7]), _k(T + 5000, 7, 2, P_AT, assists=[6])]   # tau = T - 15 s, L = T + 5 s


def _rejected_chain(f1):
    # five kills at Z, 13 s apart: one spatial sub-cluster rejected by the 60-s cap (15 + 52 s)
    vic = [(1, 3), (6, 8), (2, 4), (7, 9), (3, 5)]
    return [_k(f1 + 13000 * i, k, v, Z_AT) for i, (k, v) in enumerate(vic)]


@pytest.mark.parametrize("f1,n_foreign,isolated", [(T + 5000, 1, 0), (T + 5001, 0, 1)])
def test_isolation_foreign_kill_at_L_in_L_plus_1_out(f1, n_foreign, isolated):
    recs, dg = _v4(_a_kills() + _rejected_chain(f1))
    assert dg["rejected_max_duration"] == 1 and len(recs) == 1
    a = recs[0]
    assert (a["tau"], a["last_kill_ts"], a["n_kills"]) == (T - 15000, T + 5000, 2)
    assert (a["n_foreign_kills"], a["n_other_tau"], a["isolated"]) == (n_foreign, 0, isolated)


@pytest.mark.parametrize("fk2,n_tau,isolated", [(T + 20000, 1, 0), (T + 20001, 0, 1)])
def test_isolation_other_tau_at_L_in_L_plus_1_out(fk2, n_tau, isolated):
    recs, _ = _v4(_a_kills() + [_k(fk2, 3, 8, Q_AT)])
    assert len(recs) == 2
    a = recs[0]
    assert recs[1]["tau"] == fk2 - 15000
    assert (a["n_foreign_kills"], a["n_other_tau"], a["isolated"]) == (0, n_tau, isolated)


def test_isolation_counts_unit():
    kts = np.array([100, 200, 200, 300, 301], dtype=np.int64)
    assert isolation_counts(100, 300, [0, 1], kts, []) == (2, 0)      # kills 2 and 3 are foreign
    assert isolation_counts(100, 300, [0, 1, 2, 3], kts, [300]) == (0, 1)
    assert isolation_counts(100, 300, [0, 1, 2, 3], kts, [99, 301]) == (0, 0)


# ------------------------------------------------------------------ merge (A6)
def test_merge_with_most_recent_qualifying_candidate():
    ev = [_k(T, 6, 1, P_AT, assists=[7]), _k(T + 10000, 2, 8, Q_AT), _k(T + 24000, 3, 9, P_AT, assists=[10])]
    recs, dg = _v4(ev)
    assert dg["accepted"] == 3 and dg["merged_candidates"] == 1 and len(recs) == 2
    ac = next(r for r in recs if r["tau"] == T - 15000)
    assert ac["n_segments"] == 2 and ac["kill_ts"] == [T, T + 24000] and ac["last_kill_ts"] == T + 24000
    assert ac["parts"] == [1, 3, 6, 7, 9, 10] and (ac["n_blue"], ac["n_red"], ac["cohort"]) == (2, 4, "ASYM")
    # the R2 copy compares with the immediately preceding candidate only: no merge
    _, dg2 = detect_engagements_exact(_pack(ev), TM, PR, "r2_repro")
    assert dg2["merged_candidates"] == 0


@pytest.mark.parametrize("dt,n_final", [(45_000, 1), (45_001, 2)])
def test_merge_cap_on_unfilled_window(dt, n_final):
    ev = [_k(T, 6, 1, P_AT), _k(T + dt, 7, 2, P_AT)]
    recs, _ = _v4(ev)
    assert len(recs) == n_final
    if n_final == 1:
        assert recs[0]["last_kill_ts"] - recs[0]["tau"] == 60_000
    # horizon-filled span (R2 rule) is 80 s: never merged
    _, dg2 = detect_engagements_exact(_pack(ev), TM, PR, "r2_repro")
    assert dg2["merged_candidates"] == 0


@pytest.mark.parametrize("dt,n_final", [(50_000, 1), (50_001, 2)])
def test_merge_gap_boundary(dt, n_final):
    # a.horizon_end = tau + 35 s = T + 20 s; b.tau = T + dt - 15 s -> gap = dt - 35 s
    p = ExactParams(13700, 4264.0, max_duration_ms=100_000)
    recs, _ = _v4([_k(T, 6, 1, P_AT), _k(T + dt, 7, 2, P_AT)], params=p)
    assert len(recs) == n_final


def test_merge_requires_place():
    recs, dg = _v4([_k(T, 6, 1, P_AT), _k(T + 20000, 7, 2, Q_AT)])
    assert len(recs) == 2 and dg["merged_candidates"] == 0


def test_no_overlap_drop_in_v4():
    # two simultaneous far-apart fights: R2 drops / replaces one, v4 keeps both (not isolated)
    ev = [_k(T, 6, 1, P_AT), _k(T + 1000, 2, 7, Q_AT)]
    recs, _ = _v4(ev)
    assert len(recs) == 2 and all(r["isolated"] == 0 for r in recs)
    r2, dg2 = detect_engagements_exact(_pack(ev), TM, PR, "r2_repro")
    assert len(r2) == 1 and dg2["postmerge_overlap_dropped"] + dg2["postmerge_overlap_clipped"] == 1


# ------------------------------------------------------------------ alive gate and guards
@pytest.mark.parametrize("fk2,accepted", [(T + 24_999, False), (T + 25_000, True)])
def test_alive_gate_event_survival_boundary(fk2, accepted):
    # blue 1-4 die at T (level 1, 10:00 -> 10 s timer, dead in [T, T + 10 s)); tau2 = fk2 - 15 s
    ev = [_k(T, 6, v, Z_AT) for v in (1, 2, 3, 4)] + [_k(fk2, 7, 5, P_AT)]
    recs, dg = _v4(ev)
    assert dg["rejected_too_few_per_team"] == (0 if accepted else 1)
    assert len(recs) == (2 if accepted else 1)
    if accepted:
        assert (recs[1]["alive_blue"], recs[1]["alive_red"]) == (5, 5)


def test_time_guards_order_and_counts():
    ev = [_k(130_000, 6, 1, P_AT),                  # tau 115 s: start offset (< 120 s)
          _k(40_000, 6, 2, Q_AT),                   # tau 25 s: context (tau - 30 s < 0) is checked first
          _k(39 * 60000 - 19_999, 6, 3, P_AT)]      # tau + 35 s > last frame: horizon
    recs, dg = _v4(ev)
    assert recs == []
    assert (dg["rejected_startctx"], dg["rejected_start_offset"], dg["rejected_horizon"]) == (1, 1, 1)


class _GuardedPack(dict):
    FORBIDDEN = ("xy_raw_minute", "node_minute", "global_minute", "gold_team_minute")

    def __getitem__(self, k):
        if k in self.FORBIDDEN:
            raise AssertionError(f"forbidden pack access: {k}")
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k in self.FORBIDDEN:
            raise AssertionError(f"forbidden pack access: {k}")
        return super().get(k, default)


def _guard(monkeypatch):
    from gameplay import fight_clustering as FC
    from gameplay import fights as F

    def boom(*a, **k):
        raise AssertionError("5-s grid built")
    monkeypatch.setattr(FC, "build_5s_position_grid", boom)
    monkeypatch.setattr(F, "_build_5s_position_grid", boom)
    monkeypatch.setattr(F, "_build_5s_position_grid_impl", boom)


def test_guard_no_xy_node_or_grid_synthetic(monkeypatch):
    _guard(monkeypatch)
    ev = [_k(T, 6, 1, P_AT, assists=[7]), _k(T + 10000, 2, 8, Q_AT), _k(T + 24000, 3, 9, P_AT)]
    pk = _GuardedPack(_pack(ev))
    pk["xy_raw_minute"] = pk["node_minute"] = None
    for mode in ("v4", "r2_repro"):
        recs, _ = detect_engagements_exact(pk, TM, PR, mode)
        assert recs


def test_guard_no_xy_node_or_grid_real(monkeypatch):
    mid = "KR_7726627001"
    pk = load_exact_pack(mid, allowed_patches=["15.14"])
    if pk is None:
        pytest.skip("cache not available")
    assert set(pk) == {"meta", "events", "minute_ts"}
    _guard(monkeypatch)
    gp = _GuardedPack(pk)
    gp["xy_raw_minute"] = gp["node_minute"] = np.zeros(1)
    for mode, p in (("v4", ExactParams(14000, 4300.0)), ("r2_repro", PR)):
        recs, _ = detect_engagements_exact(gp, gp["meta"]["team_map"], p, mode)
        assert recs


def test_loader_refuses_other_patches_before_reading_events(tmp_path):
    (tmp_path / "X.meta.json").write_text(json.dumps({"patch": "15.15", "team_map": {}}), encoding="utf-8")
    (tmp_path / "X.events.json").write_text("not json", encoding="utf-8")
    np.savez(tmp_path / "X.npz", minute_ts=np.arange(3))
    assert load_exact_pack("X", tmp_path, allowed_patches=["15.14"]) is None


# ------------------------------------------------------------------ real 15.14 data
def _r2_mids():
    f = BASE / "matches_r2_1500.txt"
    if not f.exists() or not CACHE.exists():
        pytest.skip("R2 baseline or cache not available")
    return [m.strip() for m in f.read_text(encoding="utf-8").splitlines() if m.strip()]


def test_v4_invariants_on_real_matches():
    mids = _r2_mids()[:40]
    p = ExactParams(14000, 4300.0)
    n = 0
    for mid in mids:
        pk = load_exact_pack(mid, allowed_patches=["15.14"])
        assert pk is not None
        tm = pk["meta"]["team_map"]
        recs, dg = detect_engagements_exact(pk, tm, p, "v4")
        n += len(recs)
        assert dg["accepted"] - dg["merged_candidates"] == len(recs)
        seen = set()
        iv = death_intervals(pk["events"], "15.14")
        blue, red = team_sets(tm)
        for r in recs:
            assert not (seen & set(r["kill_idx"]))            # every kill in at most one engagement
            seen |= set(r["kill_idx"])
            assert r["last_kill_ts"] - r["tau"] <= p.max_duration_ms
            assert r["tau"] < r["first_kill_ts"] == min(r["kill_ts"]) and r["last_kill_ts"] == max(r["kill_ts"])
            a = event_alive(iv, r["tau"])                    # full-event intervals give the same alive
            assert (int(a[[x - 1 for x in blue]].sum()), int(a[[x - 1 for x in red]].sum())) == (r["alive_blue"], r["alive_red"])
            assert r["cohort"] == cohort_label(len(r["blue_parts"]), len(r["red_parts"]))
            assert r["isolated"] == int(r["n_foreign_kills"] == 0 and r["n_other_tau"] == 0)
    assert n > 0


def test_r2_reproduction_1500():
    mids = _r2_mids()
    assert len(mids) == 1500
    ref = []
    with open(BASE / "finals_r2_kill_full.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ref.append(tuple(r[c] for c in R2_COMPARE_COLUMNS))
    diag_ref = json.loads((BASE / "r2_results.json").read_text(encoding="utf-8"))["diag_sums"]["r2_kill"]
    mine, diag = [], Counter()
    for mid in mids:
        pk = load_exact_pack(mid, allowed_patches=["15.14"])
        assert pk is not None, mid
        recs, dg = detect_engagements_exact(pk, pk["meta"]["team_map"], PR, "r2_repro")
        mine.extend(r2_tsv_row(r) for r in recs)
        diag.update({k: v for k, v in dg.items() if k in EP.R2_DIAG_KEYS})
    assert len(ref) == 29421 and len({r[:4] for r in ref}) == 29420   # one duplicate (mid, fk, killer, victim) key
    assert mine == ref                                                  # row for row, 11 decision columns
    for k in EP.R2_DIAG_KEYS:
        assert int(diag.get(k, 0)) == int(diag_ref.get(k, 0)), k
    assert (diag["accepted"], diag["merged_candidates"], diag["postmerge_overlap_dropped"], diag["postmerge_replaced"],
            diag["postmerge_removed"], diag["postmerge_overlap_clipped"]) == (41564, 1564, 6540, 1711, 2328, 3607)
