"""Tests for validation A: weighting, calibration arithmetic, banding, missingness, CI."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_temporal_winprob_claude import (  # noqa: E402
    AGE_BANDS, POINTS, TIME_BANDS, band_of, decompose, match_bootstrap, match_weights,
    metrics, reliability,
)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs/temporal_winprob_v2"


def test_match_weights_equalise_match_totals():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    w = match_weights(groups)
    totals = {m: w[groups == m].sum() for m in ("a", "b", "c")}
    assert pytest.approx(totals["a"]) == totals["b"] == totals["c"]
    assert pytest.approx(w.mean()) == 1.0


def test_reliability_shares_sum_to_one_and_ece_zero_when_calibrated():
    # Ten bins, each with observed frequency exactly equal to the predicted value.
    p = np.repeat(np.linspace(.05, .95, 10), 100)
    rng = np.random.default_rng(0)
    y = (rng.random(len(p)) < p).astype(int)
    # Force exact calibration bin by bin so the assertion is about arithmetic, not sampling.
    for lo in np.unique(p):
        mask = p == lo
        y[mask] = 0
        y[np.flatnonzero(mask)[:int(round(lo * mask.sum()))]] = 1
    w = np.ones(len(p))
    rows, ece = reliability(y, p, w)
    assert pytest.approx(sum(r["weight_share"] for r in rows)) == 1.0
    assert ece < 1e-9


def test_reliability_detects_overconfidence():
    p = np.full(200, .9)
    y = np.zeros(200, dtype=int)
    y[:100] = 1                       # predicted .9, observed .5
    rows, ece = reliability(y, p, np.ones(200))
    hot = [r for r in rows if r["n"]][0]
    assert hot["gap"] == pytest.approx(-0.4)
    assert ece == pytest.approx(0.4)


def test_metrics_reports_na_auc_for_single_class_without_crashing():
    y = np.ones(50, dtype=int)
    p = np.full(50, .8)
    got = metrics(y, p, np.array([f"m{i//5}" for i in range(50)]))
    assert got["auc"] is None
    assert "single class" in got["auc_na_reason"]
    assert np.isfinite(got["log_loss"]) and np.isfinite(got["brier"])
    assert got["n"] == 50 and got["matches"] == 10


def test_band_of_edges_are_half_open_and_catch_all_upper():
    values = np.array([2.0, 9.999, 10.0, 29.999, 30.0, 120.0, np.nan])
    bands = band_of(values, TIME_BANDS)
    assert list(bands[:6]) == ["2-10", "2-10", "10-20", "20-30", "30+", "30+"]
    assert bands[6] == ""                     # NaN belongs to no band

    ages = band_of(np.array([0.0, 15.0, 44.9, 45.0, 60.04]), AGE_BANDS)
    assert list(ages) == ["0-15", "15-30", "30-45", "45+", "45+"]


def test_decompose_partitions_rows_without_double_counting():
    rng = np.random.default_rng(7)
    n = 400
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    g = np.array([f"m{i//4}" for i in range(n)])
    minute = rng.uniform(2, 45, n)
    table = decompose(y, p, g, minute, TIME_BANDS, "time")
    assert sum(cell["n"] for cell in table.values()) == n


def test_bootstrap_is_deterministic_and_declines_tiny_groups():
    rng = np.random.default_rng(1)
    n = 600
    g = np.array([f"m{i//3}" for i in range(n)])
    y = rng.integers(0, 2, n)
    p_by = {"maymin": rng.random(n), "expanded": rng.random(n)}
    a = match_bootstrap(y, p_by, g, replicates=25)
    b = match_bootstrap(y, p_by, g, replicates=25)
    assert a == b
    assert a["families"]["maymin"]["brier"]["lo"] < a["families"]["maymin"]["brier"]["hi"]
    assert "excludes_zero" in a["expanded_minus_maymin"]["brier"]

    tiny = np.array([f"m{i//3}" for i in range(30)])       # 10 matches < MIN_MATCHES_FOR_CI
    declined = match_bootstrap(y[:30], {k: v[:30] for k, v in p_by.items()}, tiny)
    assert "na_reason" in declined


def test_bootstrap_duplicated_match_keeps_its_own_weight():
    # Two matches with very different row counts: the CI must not be dominated by row count.
    g = np.array(["a"] * 90 + ["b"] * 10 + [f"m{i}" for i in range(40)])
    y = np.array([1] * 90 + [0] * 10 + [i % 2 for i in range(40)])
    p = np.where(y == 1, .8, .2)
    out = match_bootstrap(y, {"maymin": p}, g, replicates=20)
    assert out["matches"] == 42
    assert out["families"]["maymin"]["brier"]["replicates"] > 0


@pytest.mark.skipif(not (V2 / "engagement_changes.npz").exists(), reason="frozen v2 artifacts absent")
def test_missing_followups_match_the_frozen_v2_denominators():
    """+30s / +60s queries past game end are dropped, not imputed or carried forward."""
    with np.load(V2 / "engagement_changes.npz", allow_pickle=False) as z:
        snapshot = z["snapshot_ms"]
        query = z["query_ms"]
        expanded = z["expanded"]
    valid = np.isfinite(expanded) & (snapshot >= 0) & (query >= 0)
    stored = json.loads((V2 / "results.json").read_text(encoding="utf-8"))["changes"]["expanded"]
    assert valid.shape[1] == len(POINTS)
    assert int(valid[:, 0].sum()) == int(valid[:, 1].sum()) == stored["last_kill_proxy"]["n"]
    assert int(valid[:, 2].sum()) == stored["plus_30s"]["n"]
    assert int(valid[:, 3].sum()) == stored["plus_60s"]["n"]
    # Missingness is monotone in the horizon: nothing reappears after dropping out.
    assert np.all(valid[:, 3] <= valid[:, 2])


@pytest.mark.skipif(not (V2 / "engagement_changes.npz").exists(), reason="frozen v2 artifacts absent")
def test_query_times_are_ordered_and_snapshots_never_come_from_the_future():
    with np.load(V2 / "engagement_changes.npz", allow_pickle=False) as z:
        query, snapshot = z["query_ms"], z["snapshot_ms"]
    observed = snapshot >= 0
    assert np.all(snapshot[observed] <= query[observed])          # no future interpolation
    age_s = (query - snapshot)[observed] / 1000.
    # Source frames arrive about every 60s with jitter, so one frame of staleness is the cap.
    assert age_s.min() >= 0 and age_s.max() <= 61.0
    # The 60.04s maximum quoted in TEMPORAL_WINPROB_RESULTS.md covers only pre and end; the
    # follow-up queries reach slightly further, so the documented figure is not a global bound.
    pre_end = (query[:, :2] - snapshot[:, :2])[snapshot[:, :2] >= 0] / 1000.
    assert pre_end.max() == pytest.approx(60.04, abs=.01)
    assert age_s.max() > pre_end.max()
    ordered = query[:, 1] > query[:, 0]
    assert ordered.all()                                          # end strictly after pre


# --------------------------------------------------------------- validation B

from scripts.validate_boundary_sensitivity_claude import (  # noqa: E402
    NEUTRAL_BANDS, describe, scale_class,
)


def test_scale_class_never_puts_the_unknown_sentinel_in_pick():
    """-1 means 'participants unknown', not 'one participant'; it must stay separate."""
    classes = scale_class(np.array([-1, 0, 1, 2, 3, 4, 5]))
    assert list(classes) == ["unknown(-1)", "pick(<=1)", "pick(<=1)", "skirmish(2-3)",
                             "skirmish(2-3)", "teamfight(>=4)", "teamfight(>=4)"]
    assert (classes == "pick(<=1)").sum() == 2          # 0 and 1 only, never -1


def test_describe_sign_fractions_partition_and_match_weighting_differs():
    delta = np.array([.10, -.05, .02, -.02, .00])
    groups = np.array(["a", "a", "a", "a", "b"])        # match a dominates the raw mean
    got = describe(delta, groups)
    assert got["blue_increase_fraction"] + got["red_increase_fraction"] + got["exact_zero_fraction"] == 1.0
    assert got["n"] == 5 and got["matches"] == 2
    assert got["mean_pp_unweighted"] != got["mean_pp_match_weighted"]
    assert got["median_abs_pp"] == pytest.approx(2.0)


def test_neutral_band_share_is_monotone_in_the_threshold():
    rng = np.random.default_rng(3)
    delta = rng.normal(0, .08, 5000)
    shares = [float((np.abs(delta) <= t).mean()) for t in NEUTRAL_BANDS]
    assert shares == sorted(shares)
    assert shares[0] == 0.0                            # exact zeros do not occur in practice


@pytest.mark.skipif(not (V2 / "engagement_changes.npz").exists(), reason="frozen v2 artifacts absent")
def test_immediate_delta_reproduces_the_frozen_v2_headline():
    """Validation B must not silently redefine the delta it is characterising."""
    with np.load(V2 / "engagement_changes.npz", allow_pickle=False) as z:
        recomputed = z["expanded"][:, 1] - z["expanded"][:, 0]
        stored_delta = z["expanded_delta"][:, 0]
    assert np.allclose(recomputed, stored_delta, equal_nan=True)
    stored = json.loads((V2 / "results.json").read_text(encoding="utf-8"))["changes"]["expanded"]
    ok = np.isfinite(recomputed)
    assert float(np.mean(recomputed[ok]) * 100) == pytest.approx(
        stored["last_kill_proxy"]["mean_percentage_points"], abs=1e-9)


# --------------------------------------------------------------- validation C

from scripts.validate_objective_passthrough_claude import objective_events  # noqa: E402

TEAM_MAP = {i: (100 if i <= 5 else 200) for i in range(1, 11)}


def _pack(events):
    return {"events": events}


def test_objective_events_separates_elder_from_elemental_dragons():
    got = objective_events(_pack([
        {"type": "ELITE_MONSTER_KILL", "timestamp": 100, "killerTeamId": 100,
         "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON"},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 200, "killerTeamId": 200,
         "monsterType": "DRAGON", "monsterSubType": "ELDER_DRAGON"},
    ]), TEAM_MAP)
    assert got == [(100, 100, "dragon", "FIRE"), (200, 200, "elder", "DRAGON")]


def test_objective_events_maps_soul_aliases_and_falls_back_to_the_team_map():
    got = objective_events(_pack([
        {"type": "DRAGON_SOUL_GIVEN", "timestamp": 300, "teamId": 100, "dragonSoul": "Infernal"},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 400, "killerId": 7,
         "monsterType": "BARON_NASHOR"},          # no killerTeamId -> resolve via team map
    ]), TEAM_MAP)
    assert got == [(300, 100, "soul", "FIRE"), (400, 200, "baron", "BARON_NASHOR")]


def test_objective_events_flags_unresolvable_teams_instead_of_guessing():
    got = objective_events(_pack([
        {"type": "ELITE_MONSTER_KILL", "timestamp": 500, "killerId": 0, "monsterType": "HORDE"},
    ]), {})
    assert got == [(500, 0, "unknown_team", "HORDE")]


def test_objective_events_are_sorted_and_ignore_unrelated_types():
    got = objective_events(_pack([
        {"type": "ELITE_MONSTER_KILL", "timestamp": 900, "killerTeamId": 100, "monsterType": "RIFTHERALD"},
        {"type": "CHAMPION_KILL", "timestamp": 800, "killerId": 1},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 700, "killerTeamId": 200, "monsterType": "ATAKHAN"},
    ]), TEAM_MAP)
    assert [g[0] for g in got] == [700, 900]
    assert [g[2] for g in got] == ["atakhan", "herald"]


VAL = ROOT / "outputs/temporal_winprob_claude_validation/results_c.json"


@pytest.mark.skipif(not VAL.exists(), reason="validation C has not been run")
def test_source_events_reproduce_the_frozen_objective_window_counts():
    got = json.loads(VAL.read_text(encoding="utf-8"))["C1_event_census"]
    assert got["reproduction_of_frozen_immediate_counts"]["matches"] == {
        "baron": True, "dragons": True, "elder": True, "soul_event_recorded": True}
    assert got["matches_missing_from_cache"] == []
