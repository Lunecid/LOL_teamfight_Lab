import numpy as np
import pytest

from analysis.temporal_boundary import (
    bandwidth_sweep,
    cluster_bootstrap,
    consecutive_intervals,
    episode_counts,
    kde_valley,
    mixture_crossing,
    plateau_ari,
    stratified_valleys,
    temporal_clusters,
    to_log,
)


def _synthetic_logdt(n: int, seed: int = 0, m1: float = 5.0, m2: float = 60.0, w1: float = 0.55) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = rng.random(n) < w1
    x = np.where(k, rng.normal(np.log10(m1), 0.30, n), rng.normal(np.log10(m2), 0.35, n))
    return x


class TestEstimators:
    def test_kde_valley_recovers_bimodal_structure(self):
        est = kde_valley(_synthetic_logdt(60_000))
        assert est.ok
        assert 3.5 < est.mode1_s < 7.5
        assert 40 < est.mode2_s < 90
        assert 10 < est.valley_s < 25
        assert est.depth < 0.9

    def test_kde_valley_unimodal_is_flagged(self):
        x = np.random.default_rng(1).normal(1.0, 0.3, 20_000)
        est = kde_valley(x)
        assert not est.ok
        assert np.isnan(est.valley_s)

    def test_mixture_crossing_agrees_with_valley(self):
        x = _synthetic_logdt(60_000)
        est = kde_valley(x)
        mix = mixture_crossing(x)
        assert mix["ok"]
        assert abs(np.log10(mix["crossing_s"]) - np.log10(est.valley_s)) < 0.2

    def test_bandwidth_sweep_is_stable(self):
        sweep = bandwidth_sweep(_synthetic_logdt(60_000), bandwidths=(0.05, 0.08, 0.12))
        valleys = [v["valley_s"] for v in sweep.values() if v["ok"]]
        assert len(valleys) == 3
        assert max(valleys) / min(valleys) < 1.5


class TestClustering:
    def test_consecutive_intervals(self):
        assert consecutive_intervals([10.0, 12.0, 12.0, 30.0]).tolist() == [2.0, 18.0]
        assert consecutive_intervals([5.0]).size == 0

    def test_temporal_clusters_single_linkage(self):
        ts = [0, 5, 10, 40, 45, 100]
        assert temporal_clusters(ts, 18).tolist() == [0, 0, 0, 1, 1, 2]
        assert temporal_clusters(ts, 31).tolist() == [0, 0, 0, 0, 0, 1]
        assert temporal_clusters([], 18).size == 0

    def test_temporal_clusters_unsorted_input(self):
        assert temporal_clusters([40, 0, 5], 18).tolist() == [1, 0, 0]

    def test_plateau_ari_reference_is_one(self):
        per_match = [[0, 5, 10, 40, 45, 100], [0, 3, 60, 63, 130]]
        ari = plateau_ari(per_match, gaps_s=[18, 30], ref_gap_s=18)
        assert ari["18"] == pytest.approx(1.0)
        assert ari["30"] < 1.0

    def test_episode_counts(self):
        counts = episode_counts([[0, 5, 10, 40, 45, 100]], gaps_s=[18, 100])
        assert counts["18"] == 3.0 and counts["100"] == 1.0


class TestUncertainty:
    def test_cluster_bootstrap_brackets_point_estimate(self):
        rng = np.random.default_rng(3)
        per_match = [_synthetic_logdt(int(rng.integers(20, 80)), seed=i) for i in range(300)]
        point = kde_valley(np.concatenate(per_match))
        boot = cluster_bootstrap(per_match, n_boot=30, seed=1)
        assert boot["ok"] and boot["valley_found_share"] > 0.9
        assert boot["valley_s"]["p2.5"] <= point.valley_s <= boot["valley_s"]["p97.5"]

    def test_stratified_valleys(self):
        recs = [{"log_dt": _synthetic_logdt(3000, seed=i), "patch": "15.14" if i % 2 else "15.15"} for i in range(20)]
        out = stratified_valleys(recs, key=lambda r: r["patch"], min_pairs=1000)
        assert set(out) == {"15.14", "15.15"}
        assert all(v["ok"] and v["enough"] for v in out.values())

    def test_to_log_drops_nonpositive(self):
        assert to_log(np.array([0.0, -1.0, 10.0])).tolist() == [1.0]
