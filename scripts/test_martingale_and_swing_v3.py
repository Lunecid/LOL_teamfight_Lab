"""If the fight outcome is predictable, is the win-probability swing predictable too?

The intuition: at 50/50, Blue winning the fight takes it to 70/30 and Red winning takes it to
30/70, so a predictable fight outcome should make the swing predictable.  The intuition is right
about the MECHANISM and wrong about what it implies, because a calibrated win-probability model
is a martingale:

    E[ V(S_end) | S_pre ] = E[ P(blue wins match | S_end) | S_pre ] = P(blue wins match | S_pre) = V(S_pre)

so E[delta-V | anything known at the cutoff] = 0, exactly.  V(S_pre) has ALREADY priced the fight
it is about to see.  A predictable fight outcome therefore cannot show up as a predictable MEAN
swing; it must show up as an ASYMMETRY between the winning swing and the losing swing:

    q * M_plus + (1 - q) * M_minus = 0     =>     q = -M_minus / (M_plus - M_minus)

where q is the probability Blue wins the fight, M_plus the swing when Blue wins and M_minus (negative)
the swing when Red wins.  Being favoured to win a fight means winning it gains little and losing it
costs a lot.  Three things are tested here:

  T1 martingale        is E[delta-V] zero overall and inside every stratum of V_pre?
  T2 swing asymmetry   does the implied q from the swings match the realised fight-win rate?
  T3 exploitability    does the fight predictor p-hat know anything the value model does not?
                       Regress delta-V on p-hat: a non-zero slope means V is leaving money on the
                       table and delta-V IS predictable in the mean; a zero slope means the
                       predictable part of the fight was already inside V(S_pre).
  T4 variance          how much of Var(delta-V) is direction (unpredictable at the mean) versus
                       magnitude (predictable - it is headroom)?

Everything is by game-time band as well as pooled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import match_weights

HORIZONS = {"end": 1, "plus_30s": 2, "plus_60s": 3}
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))


def match_ci(values, groups, n_boot=400, seed=7):
    """Match-level bootstrap CI for a weighted mean."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(len(uniq), size=len(uniq))])
        draws.append(float(values[idx].mean()))
    a = np.asarray(draws)
    return {"mean": float(values.mean()), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)),
            "excludes_zero": bool(np.percentile(a, 2.5) > 0 or np.percentile(a, 97.5) < 0)}


def swing_table(delta, won, groups, p_hat=None):
    """M+, M-, the fight-win rate the swings imply, and the realised one."""
    if won.sum() < 20 or (~won).sum() < 20:
        return {"n": int(len(delta)), "note": "too few of one outcome"}
    m_plus, m_minus = float(delta[won].mean()), float(delta[~won].mean())
    q_real = float(won.mean())
    q_implied = float(-m_minus / (m_plus - m_minus)) if m_plus != m_minus else float("nan")
    cell = {"n": int(len(delta)), "M_plus_pp": m_plus, "M_minus_pp": m_minus,
            "q_realised": q_real, "q_implied_by_swings": q_implied,
            "implied_minus_realised": q_implied - q_real,
            "mean_delta_pp": match_ci(delta, groups)}
    if p_hat is not None:
        cell["mean_p_hat"] = float(p_hat.mean())
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/engagement_predictor_v3")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/engagement_predictor_v3/martingale_swing.json")
    a = ap.parse_args()

    with np.load(a.pred_dir / "predictions.npz", allow_pickle=False) as z:
        ids = z["test_id"].astype(str)
        groups = z["test_match"].astype(str)
        market = z["test_market"]
        p_hat = z["test_p_market"]
        minute = z["test_minute"]
    with np.load(a.v3_dir / "engagement_changes.npz", allow_pickle=False) as z:
        order = {e: i for i, e in enumerate(z["id"].astype(str))}
        rows = [order[e] for e in ids]
        V = z["expanded"][rows]                              # pre, end, +30s, +60s
    v_pre = V[:, 0]
    headroom = np.minimum(v_pre, 1 - v_pre)
    lab = market >= 0
    out = {"rows": int(len(ids)), "labelled": int(lab.sum()),
           "identity": "q = -M_minus / (M_plus - M_minus) holds exactly when E[delta-V]=0",
           "horizons": {}}

    for hname, hcol in HORIZONS.items():
        d = (V[:, hcol] - v_pre) * 100.0
        ok = lab & np.isfinite(d)
        dd, ww, gg, pp, vv, mm = d[ok], market[ok] == 1, groups[ok], p_hat[ok], v_pre[ok], minute[ok]
        cell = {"n": int(ok.sum()),
                "T1_martingale_overall": match_ci(dd, gg),
                "T1_by_V_pre_decile": {}, "T2_by_p_hat_decile": {}, "T3_exploitability": {},
                "T4_variance": {}, "by_game_time": {}}

        edges = np.quantile(vv, np.linspace(0, 1, 11))
        for i in range(10):
            m = (vv >= edges[i]) & (vv <= edges[i + 1] if i == 9 else vv < edges[i + 1])
            if m.sum() < 100:
                continue
            cell["T1_by_V_pre_decile"][f"D{i+1} V_pre[{edges[i]:.3f},{edges[i+1]:.3f}]"] = {
                **swing_table(dd[m], ww[m], gg[m], pp[m]), "mean_V_pre": float(vv[m].mean())}

        pe = np.quantile(pp, np.linspace(0, 1, 11))
        for i in range(10):
            m = (pp >= pe[i]) & (pp <= pe[i + 1] if i == 9 else pp < pe[i + 1])
            if m.sum() < 100:
                continue
            cell["T2_by_p_hat_decile"][f"D{i+1} p[{pe[i]:.3f},{pe[i+1]:.3f}]"] = swing_table(
                dd[m], ww[m], gg[m], pp[m])

        # T3: does p-hat move the MEAN swing?  slope of delta-V on p-hat, match-weighted
        w = match_weights(gg)
        slope, intercept, r, pv, se = stats.linregress(pp, dd)
        centred = pp - pp.mean()
        wls = float(np.sum(w * centred * (dd - np.average(dd, weights=w))) / np.sum(w * centred ** 2))
        uniq, inv = np.unique(gg, return_inverse=True)
        rws = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        rng = np.random.default_rng(7)
        boots = []
        for _ in range(400):
            idx = np.concatenate([rws[i] for i in rng.integers(len(uniq), size=len(uniq))])
            c = pp[idx] - pp[idx].mean()
            boots.append(float(np.sum(c * (dd[idx] - dd[idx].mean())) / np.sum(c ** 2)))
        b = np.asarray(boots)
        cell["T3_exploitability"] = {
            "slope_pp_per_unit_p": float(slope), "slope_match_weighted": wls,
            "slope_ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "excludes_zero": bool(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0),
            "pearson_r": float(r),
            "reading": "a slope indistinguishable from zero means the predictable part of the fight "
                       "was already priced into V(S_pre); a positive slope means it was not"}

        # T4: direction versus magnitude
        sign = np.where(ww, 1.0, -1.0)
        cell["T4_variance"] = {
            "var_delta": float(dd.var()),
            "var_if_direction_known": float((np.abs(dd) * sign).var()),   # identical by construction
            "mean_abs_pp": float(np.abs(dd).mean()),
            "sd_abs_pp": float(np.abs(dd).std()),
            "spearman_abs_vs_headroom": float(stats.spearmanr(np.abs(dd), np.minimum(vv, 1 - vv)).correlation),
            "share_of_var_from_magnitude_spread": float(np.abs(dd).var() / dd.var()),
            "reading": "delta-V is (direction) x (magnitude); the magnitude tracks headroom and is "
                       "predictable, the direction is what a fight predictor would have to supply"}

        for lo, hi in TIME_BANDS:
            bm = (mm >= lo) & (mm < hi)
            if bm.sum() < 300:
                continue
            band = {"n": int(bm.sum()), "mean_V_pre": float(vv[bm].mean()),
                    "mean_headroom": float(np.minimum(vv[bm], 1 - vv[bm]).mean()),
                    "martingale": match_ci(dd[bm], gg[bm]),
                    "swings": swing_table(dd[bm], ww[bm], gg[bm], pp[bm]),
                    "mean_abs_pp": float(np.abs(dd[bm]).mean())}
            cell["by_game_time"][f"{lo}-{hi}"] = band
        out["horizons"][hname] = cell

    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    e = out["horizons"]["end"]
    print(json.dumps({"martingale_end": e["T1_martingale_overall"],
                      "exploitability_end": e["T3_exploitability"],
                      "abs_vs_headroom": e["T4_variance"]["spearman_abs_vs_headroom"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
