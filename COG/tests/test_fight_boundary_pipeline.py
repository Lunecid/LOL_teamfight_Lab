"""End-to-end test of the fight-boundary pipeline on a synthetic corpus with known structure."""
import numpy as np
import pytest

from analysis.boundary_spec import BoundarySpec
from analysis.fight_boundary_pipeline import build_spec, drift_decision, drift_markdown
from analysis.kill_pairs import MatchRecord
from analysis.map_regions import classify_points
from analysis.spatial_boundary import crossover, estimate_spatial, mass_near_boundary


def synthetic_corpus(n_matches=250, seed=0, within_s=(2, 8), between_s=(35, 120), fight_spread_u=900.0,
                     p_share_same=0.95, p_share_diff=0.08, p_simultaneous=0.25):
    """Fights = bursts of 1..6 kills, seconds apart, at one map spot; between fights tens of seconds and
    a new spot far away.  With p_simultaneous a second fight starts elsewhere while one is running, so
    consecutive kills can be seconds apart yet far apart (different fights).  Consecutive kills inside a
    burst share a champion with p_share_same, across bursts with p_share_diff."""
    rng = np.random.default_rng(seed)
    records, pairs = [], {k: [] for k in ("match_idx", "dt", "dd", "shared", "x1", "y1", "x2", "y2", "minute", "same_victim_team")}
    for mi in range(n_matches):
        t, events, fid_counter = 120.0, [], 0
        n_f = rng.integers(12, 30)
        for f in range(n_f):
            starts = [t]
            if rng.random() < p_simultaneous:
                starts.append(t + rng.uniform(0.5, 6.0))
            for st in starts:
                centre = rng.uniform(2000, 12800, 2)
                k = rng.choice([1, 2, 3, 4, 5, 6], p=[0.45, 0.22, 0.13, 0.09, 0.06, 0.05])
                tt = st
                for j in range(k):
                    if j:
                        tt += rng.uniform(*within_s)
                    events.append((tt, centre + rng.normal(0, fight_spread_u / 2, 2), fid_counter))
                fid_counter += 1
            t = max(e[0] for e in events) + rng.uniform(*between_s)
        events.sort(key=lambda e: e[0])
        ts = np.asarray([e[0] for e in events]); pos = np.asarray([e[1] for e in events]); fid = [e[2] for e in events]
        records.append(MatchRecord(f"m{mi}", "15.14", ts, len(ts)))
        for i in range(len(ts) - 1):
            same = fid[i] == fid[i + 1]
            pairs["match_idx"].append(mi); pairs["dt"].append(ts[i + 1] - ts[i])
            pairs["dd"].append(float(np.linalg.norm(pos[i + 1] - pos[i])))
            pairs["shared"].append(bool(rng.random() < (p_share_same if same else p_share_diff)))
            pairs["x1"].append(pos[i][0]); pairs["y1"].append(pos[i][1]); pairs["x2"].append(pos[i + 1][0]); pairs["y2"].append(pos[i + 1][1])
            pairs["minute"].append(ts[i] / 60); pairs["same_victim_team"].append(True)
    P = {k: np.asarray(v) for k, v in pairs.items()}
    P["region1"], P["tangent1"] = classify_points(np.stack([P["x1"], P["y1"]], 1))
    P["region2"], _ = classify_points(np.stack([P["x2"], P["y2"]], 1))
    P["patch_per_match"] = np.asarray(["15.14"] * n_matches)
    return records, P


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus()


class TestSpatial:
    def test_crossover_between_fight_spread_and_far(self, corpus):
        _, P = corpus
        win = P["dt"] <= 18
        cx, curve = crossover(P["dd"][win], P["shared"][win].astype(bool))
        assert cx is not None and 1000 < cx < 6000
        assert curve[0][1] > 0.85

    def test_estimate_spatial_reports_ci_and_regions(self, corpus):
        _, P = corpus
        S = estimate_spatial(P, gap_s=18, n_boot=20)
        assert S["ok"] and S["bootstrap"]["p2.5"] <= S["crossover_u"] <= S["bootstrap"]["p97.5"]
        assert 0 <= S["mass_near_boundary"] <= 1
        assert isinstance(S["by_region"], dict)

    def test_mass_near_boundary(self):
        dd = np.array([100, 3900, 4000, 4100, 9000.0])
        assert mass_near_boundary(dd, 4000, tol=0.05) == pytest.approx(3 / 5)


class TestPipeline:
    def test_build_spec_recovers_structure(self, corpus):
        recs, P = corpus
        spec, details = build_spec("pooled", ["15.14"], recs, P, n_boot=20)
        assert spec.gap_source == "valley" and 8 < spec.gap_s < 30
        assert spec.gap_plateau_s and spec.gap_plateau_s[0] <= spec.gap_s <= spec.gap_plateau_s[1]
        assert spec.diameter_source == "crossover" and 1000 < spec.diameter_u < 6000
        assert spec.diameter_ci_u[0] <= spec.diameter_u <= spec.diameter_ci_u[1]
        assert spec.validity_radius_u == 1800 and spec.radius_source == "default"
        ov = spec.detector_overrides()
        assert ov["TF2_KILL_CLUSTER_GAP_MS"] == round(spec.gap_s * 1000) and ov["CLUSTER_MAX_DIAMETER"] == spec.diameter_u

    def test_thin_slice_falls_back_to_pooled(self, corpus):
        recs, P = corpus
        pooled, _ = build_spec("pooled", ["15.14"], recs, P, n_boot=10)
        # a slice with fewer than 50 intervals cannot support a valley estimate
        thin_recs = [MatchRecord("m0", "15.14", recs[0].ts[:20], 20)]
        keep = np.zeros(P["dt"].shape[0], dtype=bool)
        keep[np.where(P["match_idx"] == 0)[0][:19]] = True
        thin_P = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == P["dt"].shape[:1] else v) for k, v in P.items()}
        spec, _ = build_spec("patch:x", ["x"], thin_recs, thin_P, n_boot=5, pooled=pooled)
        assert spec.gap_source == "pooled" and spec.gap_s == pooled.gap_s
        assert spec.diameter_source in ("pooled", "crossover")

    def test_drift_decision_pooled_when_identical(self, corpus):
        recs, P = corpus
        pooled, _ = build_spec("pooled", ["15.14"], recs, P, n_boot=10)
        same = BoundarySpec.from_dict({**pooled.to_dict(), "scope": "patch:15.14"})
        d = drift_decision({"15.14": same}, pooled)
        assert d["verdict"] == "pooled" and d["patches"]["15.14"]["inside"]

    def test_drift_decision_flags_shift(self, corpus):
        recs, P = corpus
        pooled, _ = build_spec("pooled", ["15.14"], recs, P, n_boot=10)
        shifted = BoundarySpec.from_dict({**pooled.to_dict(), "scope": "patch:16.1",
                                          "diameter_u": pooled.diameter_u * 1.5,
                                          "diameter_ci_u": [pooled.diameter_u * 1.45, pooled.diameter_u * 1.55]})
        d = drift_decision({"16.1": shifted}, pooled)
        assert d["verdict"] == "per_patch" and not d["patches"]["16.1"]["inside"]
        md = drift_markdown(d, {"16.1": shifted}, pooled)
        assert "per_patch" in md

    def test_spec_json_roundtrip_and_scale_classes(self, tmp_path, corpus):
        recs, P = corpus
        spec, _ = build_spec("pooled", ["15.14"], recs, P, n_boot=5)
        spec.to_json(tmp_path / "spec.json")
        back = BoundarySpec.from_json(tmp_path / "spec.json")
        assert back.to_dict() == spec.to_dict()
        cls = back.scale_class([1, 2, 3, 4, 5, -1], [5, 2, 3, 4, 5, 2])
        assert cls.tolist() == ["pick", "skirmish", "skirmish", "teamfight", "teamfight", "unknown"]
