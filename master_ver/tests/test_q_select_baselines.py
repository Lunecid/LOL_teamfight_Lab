"""q selection rules and references."""
import numpy as np
import pytest

from studies.q_production_20261005 import qbaselines, qselect


def _res(brier, keys="k", status="completed"):
    return {"status": status, "scores": {"q_dev": {"raw_match_brier": brier}}, "q_dev_keys_sha256": keys}


def test_common_card_needs_all_ten_anchor_fits_on_identical_keys():
    anchors = [f"c{i}" for i in range(10)]
    cards = ["a", "b", "c"]
    results = {(c, "a"): _res(0.20) for c in anchors} | {(c, "b"): _res(0.19) for c in anchors} | {(c, "c"): _res(0.10) for c in anchors[:9]}
    results[("c3", "b")] = _res(0.19, keys="other")
    out = qselect.select_common(results, anchors, cards)
    assert out["selected"] == "a" and not out["cards"]["b"]["eligible"] and not out["cards"]["c"]["eligible"]
    assert np.isclose(out["cards"]["a"]["score"], 0.20)
    assert not qselect.select_common({}, anchors, cards)["complete"]


def test_within_family_ties_go_to_card_order_and_failures_are_skipped():
    out = qselect.select_within({"x": _res(0.2), "y": _res(0.2), "z": {"status": "failed"}}, ["z", "y", "x"])
    assert out["selected"] == "y" and not out["complete"]


def test_family_selection_tolerance_order_and_logistic_terminal_failure():
    reps = {f: _res(0.25) for f in qselect.FAMILY_ORDER}
    reps["logistic"] = None
    reps["relational_mpnn"] = _res(0.2400)
    reps["graphsage"] = _res(0.2404)            # within 0.0005 of the best -> tied, earlier in the order
    out = qselect.select_family(reps, logistic_attempts=[{"status": "failed", "reason": "nonconverged"}])
    assert out["selected"] == "graphsage" and out["complete"] and out["logistic_terminal_nonconvergence"]
    assert not qselect.select_family(reps)["complete"]                     # logistic missing without a recorded attempt
    reps.pop("ft_transformer")                                              # removed before any fit (D-Q15): not required
    out = qselect.select_family(reps, logistic_attempts=[{"status": "failed", "reason": "nonconverged"}])
    assert out["complete"] and "ft_transformer" not in out["briers"] and out["excluded_families"] == qselect.EXCLUDED_FAMILIES


def test_references_fit_and_pt_flex_has_univariate_and_tensor_columns():
    rng = np.random.default_rng(0)
    n = 2000
    p, t = rng.uniform(0.05, 0.95, n), rng.uniform(2, 40, n)
    y = (rng.random(n) < p).astype(float)
    train = {"p": p, "t": t, "y": y, "w_norm": np.ones(n)}
    for name, card in (("time_only", {"C": 1.0}), ("p_only", {"C": 1.0}), ("PT_linear", {"C": 1.0}),
                       ("PT_flex", {"C": 1.0, "p_knots": 6, "time_knots": 4, "degree": 3})):
        model, design = qbaselines.fit_reference(name, card, train)
        x = design(p, t)
        logits = model.predict_logits(x)
        assert np.isfinite(logits).all()
        if name == "PT_flex":
            bp = design.spline_p.transform(qbaselines.logit_p(p).reshape(-1, 1)).shape[1]
            bt = design.spline_t.transform(t.reshape(-1, 1)).shape[1]
            assert x.shape[1] == bp + bt + bp * bt
    assert np.isclose(qbaselines.constant_rate(y, np.ones(n)), y.mean())


def test_degenerate_knots_fail_the_card():
    p = np.full(500, 0.5)
    with pytest.raises(ValueError):
        qbaselines.Design("PT_flex", {"C": 1.0, "p_knots": 6, "time_knots": 4, "degree": 3}, p, np.linspace(2, 40, 500))
