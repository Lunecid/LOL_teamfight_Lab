"""gameplay.respawn_rules / gameplay.event_survival (v4-exact stage 1, task A1)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gameplay import respawn_rules as RR
from gameplay.event_survival import death_intervals, event_alive, respawn_remaining_s, respawned_since

WT = Path(__file__).resolve().parents[1]
DIAG = WT.parents[1] / "outputs" / "diag_survival_dbscan_20260925"
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
SRC_SHA = "ba34b408fb2b3af6133252bec97f39911537c01d4d1d3691c729c956cf7157f0"


def _kill(ts, victim):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "victimId": victim, "killerId": 1}


def _lvl(ts, pid, level):
    return {"type": "LEVEL_UP", "timestamp": ts, "participantId": pid, "level": level}


# ------------------------------------------------------------------ respawn_rules
def test_respawn_rules_is_verbatim_copy():
    lines = (WT / "gameplay" / "respawn_rules.py").read_bytes().replace(b"\r\n", b"\n").split(b"\n")
    assert lines[1].decode().endswith(SRC_SHA)
    body = b"\n".join(lines[3:])
    assert hashlib.sha256(body).hexdigest() == SRC_SHA
    src = DIAG / "scripts" / "respawn_rules.py"
    if src.exists():
        assert hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == SRC_SHA


def test_level_1_and_18_timers():
    assert RR.death_timer_s("15.14", 1, 0.0) == pytest.approx(10.0)
    assert RR.death_timer_s("15.14", 18, 600.0) == pytest.approx(52.5)
    assert RR.death_timer_s("15.14", 0, 0.0) == pytest.approx(10.0)  # clamped to 1
    assert RR.death_timer_s("16.15", 20, 0.0) == pytest.approx(52.5)  # 19-20 use level 18
    assert [RR.death_timer_s("15.14", lv, 0.0) for lv in range(1, 10)] == [10, 10, 12, 12, 14, 16, 20, 25, 28]


@pytest.mark.parametrize("minute,factor", [
    (0.0, 0.0), (15.0, 0.0), (15.5, 0.00425), (30.0, 0.1275), (30.5, 0.1305),
    (45.0, 0.2175), (45.5, 0.232), (60.0, 0.50), (90.0, 0.50),
])
def test_time_factor_boundaries(minute, factor):
    assert RR.time_factor(minute * 60.0) == pytest.approx(factor, abs=1e-12)


@pytest.mark.parametrize("m", [15.0, 30.0, 45.0])
def test_time_factor_continuous_at_breaks(m):
    eps = 1e-6
    assert RR.time_factor(m * 60 - eps) == pytest.approx(RR.time_factor(m * 60 + eps), abs=1e-8)


def test_time_factor_cap_50pct():
    m_cap = 45.0 + (0.5 - 0.2175) / 0.029
    assert RR.time_factor(m_cap * 60 - 1.0) < 0.5
    assert RR.time_factor(m_cap * 60 + 1.0) == 0.5
    assert RR.death_timer_s("15.14", 18, 3600.0) == pytest.approx(52.5 * 1.5)


@pytest.mark.parametrize("patch", ["14.15", "13.24", "16.17"])
def test_out_of_range_patch_raises(patch):
    with pytest.raises(KeyError):
        RR.death_timer_s(patch, 5, 100.0)
    with pytest.raises(KeyError):
        death_intervals([_kill(1000, 3)], patch)


def test_match_v5_major_above_16_refused_by_event_survival():
    # respawn_rules itself reads Match-V5 "17.1" as public 17.1 (in range); event_survival refuses it
    assert RR.death_timer_s("17.1", 1, 0.0) == 10.0
    with pytest.raises(KeyError):
        death_intervals([_kill(1000, 3)], "17.1")


@pytest.mark.parametrize("patch", ["14.16", "15.14", "15.16", "16.16"])
def test_in_range_patch_ok(patch):
    assert RR.death_timer_s(patch, 1, 0.0) == 10.0


# ------------------------------------------------------------------ event_survival (synthetic)
def test_level_from_level_up_at_or_before_kill():
    ev = [_lvl(50_000, 3, 2), _lvl(90_000, 3, 5), _lvl(90_001, 3, 9), _kill(90_000, 3)]
    iv = death_intervals(ev, "15.14")
    assert iv[3] == [[90_000, 90_000 + 14_000.0]]  # level 5 (<= kill), not 9
    assert all(iv[p] == [] for p in range(1, 11) if p != 3)


def test_timer_uses_game_time_at_death():
    t = 40 * 60_000
    iv = death_intervals([_lvl(1000, 7, 18), _kill(t, 7)], "15.14")
    f = 0.1275 + 0.006 * 10
    assert iv[7][0][1] == pytest.approx(t + 52.5 * (1 + f) * 1000)


def test_early_revive_cut_and_alive():
    ev = [_kill(100_000, 3), _kill(105_000, 3)]  # level 1: 10 s timer, second death after 5 s
    iv = death_intervals(ev, "15.14")
    assert iv[3] == [[100_000, 105_000], [105_000, 115_000.0]]
    a = lambda t: event_alive(iv, t)[2]
    assert a(99_999) == 1 and a(100_000) == 0 and a(104_999) == 0 and a(105_000) == 0
    assert a(114_999) == 0 and a(115_000) == 1
    assert event_alive(iv, 102_000).sum() == 9


def test_respawned_since_half_open():
    iv = death_intervals([_kill(100_000, 4)], "15.14")  # ends 110_000
    assert respawned_since(iv, 109_999, 110_000)[3] == 1
    assert respawned_since(iv, 110_000, 120_000)[3] == 0
    assert respawned_since(iv, 100_000, 109_999)[3] == 0
    assert respawned_since(iv, -1, 200_000).sum() == 1


def test_respawn_remaining_is_causal_with_t():
    ev = [_kill(100_000, 3), _kill(105_000, 3)]
    full = death_intervals(ev, "15.14")
    causal = death_intervals(ev, "15.14", t=102_000)
    assert respawn_remaining_s(causal, 102_000)[2] == pytest.approx(8.0)
    assert respawn_remaining_s(full, 102_000)[2] == pytest.approx(3.0)  # hindsight cut, documented
    assert respawn_remaining_s(causal, 99_000).sum() == 0
    assert respawn_remaining_s(death_intervals(ev, "15.14", t=110_000), 110_000)[2] == pytest.approx(5.0)


def test_t_cutoff_excludes_later_events():
    ev = [_lvl(10_000, 2, 3), _kill(20_000, 2), _lvl(20_001, 2, 4), _kill(40_000, 2)]
    assert death_intervals(ev, "15.14", t=19_999)[2] == []
    assert death_intervals(ev, "15.14", t=20_000)[2] == [[20_000, 32_000.0]]


def test_ignores_bad_ids_and_other_types():
    ev = [_kill(1000, 0), _kill(1000, 11), {"type": "WARD_PLACED", "timestamp": 5}, _lvl(1, 12, 5)]
    assert all(v == [] for v in death_intervals(ev, "15.14").values())


# ------------------------------------------------------------------ equality with p3 on 15.14 matches
def _load_p3():
    p3_path = DIAG / "scripts" / "p3_detection_arms.py"
    rr_path = DIAG / "scripts" / "respawn_rules.py"
    if not (p3_path.exists() and rr_path.exists()):
        return None
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("_p3_detection_arms_ref", p3_path)
        p3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p3)
        spec2 = importlib.util.spec_from_file_location("_p3_respawn_rules_ref", rr_path)
        rr = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(rr)
    finally:
        sys.path[:] = saved
    p3._S["RR"] = rr
    return p3


def _match_ids_1514(n):
    lst = DIAG / "p1" / "matches_smoke.json"
    if not lst.exists() or not CACHE.exists():
        return []
    rows = json.loads(lst.read_text(encoding="utf-8"))
    return [r["match_id"] for r in rows if str(r.get("patch")) == "15.14"][:n]


def _pack(mid):
    meta = json.loads((CACHE / f"{mid}.meta.json").read_text(encoding="utf-8"))
    if str(meta.get("patch")) != "15.14":
        return None
    events = json.loads((CACHE / f"{mid}.events.json").read_text(encoding="utf-8"))
    with np.load(CACHE / f"{mid}.npz", allow_pickle=False) as z:
        minute_ts = z["minute_ts"].astype(np.int64)
    return {"meta": meta, "events": events, "minute_ts": minute_ts}


def test_equal_to_p3_on_50_real_1514_matches():
    p3 = _load_p3()
    ids = _match_ids_1514(50)
    if p3 is None or len(ids) < 50:
        pytest.skip("diag scripts or 15.14 cache not available")
    rng = np.random.default_rng(20260925)
    n_checked = n_kills = 0
    for mid in ids:
        pack = _pack(mid)
        assert pack is not None, mid
        ref = p3.death_intervals(pack)
        mine = death_intervals(pack["events"], pack["meta"]["patch"])
        assert mine == ref, mid
        kts = sorted({int(e["timestamp"]) for e in pack["events"] if e.get("type") == "CHAMPION_KILL"})
        n_kills += len(kts)
        end = int(max(e["timestamp"] for e in pack["events"]))
        ends = [int(np.ceil(b)) for iv in ref.values() for _a, b in iv]
        ts = set(kts) | {k - 1 for k in kts} | {k + 1 for k in kts} | set(int(x) for x in pack["minute_ts"])
        ts |= set(ends) | {e - 1 for e in ends} | set(int(x) for x in rng.integers(0, end + 1, 60))
        mts = pack["minute_ts"]
        for t in sorted(ts):
            a = event_alive(mine, t)
            assert np.array_equal(a, p3.event_alive(ref, t)), (mid, t)
            j = int(np.searchsorted(mts, t, side="right")) - 1
            tf = int(mts[j]) if j >= 0 else -1
            r = respawned_since(mine, tf, t)
            assert np.array_equal(r, p3.respawned_since_frame(ref, mts, t)), (mid, t)
            # causal invariance: intervals from events <= t give the same alive / respawned vectors
            cz = death_intervals(pack["events"], "15.14", t=t)
            assert np.array_equal(event_alive(cz, t), a), (mid, t)
            assert np.array_equal(respawned_since(cz, tf, t), r), (mid, t)
            rem = respawn_remaining_s(cz, t)
            assert np.array_equal(rem > 0, a == 0), (mid, t)
            n_checked += 1
    assert n_kills > 1000 and n_checked > 5000
