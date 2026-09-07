"""Fight-boundary definition pipeline.

    corpus slice (patch)  -->  kill records + consecutive kill pairs
                          -->  temporal boundary G  (valley, CI, plateau)
                          -->  spatial boundary  D  (sharing crossover, CI, regions, lanes)
                          -->  validity radius   R  (Data Dragon coverage, optional)
                          -->  BoundarySpec (JSON)  -->  detector overrides + scale classes

Run per patch and pooled; ``drift_decision`` then states, by a rule fixed in
advance, whether one pooled definition serves every patch or the definition
must move with the patch:

* a patch's G is "inside" if it lies in the pooled plateau (the gap band whose
  clustering agrees with the pooled G at ARI >= 0.90);
* a patch's D is "inside" if it lies within +-10% of the pooled D and the
  bootstrap intervals overlap;
* a patch's R coverage (share of ability ranges within R) may differ from the
  pooled coverage by at most 2 points;
* verdict "pooled" when every patch is inside on all checks, else "per_patch".

Fallbacks are explicit: when a slice shows no valley (late-game-like
unimodal intervals, or too few matches) its G inherits the pooled G, and when
the pooled slice has none the mechanics default is used and flagged.
"""
from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional, Sequence

import numpy as np

from analysis.boundary_spec import DEFAULTS, BoundarySpec
from analysis.kill_pairs import MatchRecord
from analysis.spatial_boundary import estimate_spatial
from analysis.temporal_boundary import (bandwidth_sweep, cluster_bootstrap, kde_valley, mixture_crossing,
                                        plateau_ari)

GAPS = [6, 8, 10, 12, 13, 14, 15, 16, 18, 20, 22, 24, 27, 30, 36, 45, 60]
PLATEAU_ARI = 0.90
DIAMETER_REL_TOL = 0.10
RADIUS_COVERAGE_TOL = 0.02   # per-patch coverage of R may move by at most 2 points


def _plateau_band(ari: Dict[str, float], threshold: float) -> Optional[List[float]]:
    """Contiguous gap band (in GAPS order) around the reference where ARI >= threshold."""
    gaps = [float(k) for k in ari]
    ok = [ari[k] >= threshold for k in ari]
    if not any(ok):
        return None
    # take the contiguous run containing the maximum ARI
    i0 = int(np.argmax([ari[k] for k in ari]))
    lo = i0
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < len(ok) and ok[hi + 1]:
        hi += 1
    return [gaps[lo], gaps[hi]]


def estimate_temporal(records: Sequence[MatchRecord], n_boot: int, seed: int, bandwidth: float) -> dict:
    per_match = [r.log_dt for r in records if r.log_dt.size]
    all_log = np.concatenate(per_match) if per_match else np.empty(0)
    if all_log.size < 50:
        return {"ok": False, "n_intervals": int(all_log.size)}
    est = kde_valley(all_log, bandwidth=bandwidth, seed=seed)
    out = {"ok": bool(est.ok), "n_intervals": int(all_log.size), "kde": est.as_dict(),
           "bandwidth_sweep": bandwidth_sweep(all_log), "mixture": mixture_crossing(all_log, seed=seed),
           "bootstrap": cluster_bootstrap(per_match, n_boot=n_boot, seed=seed, bandwidth=bandwidth)}
    per_ts = [r.ts for r in records]
    if est.ok:
        out["ari_vs_valley"] = plateau_ari(per_ts, GAPS, ref_gap_s=est.valley_s)
        out["plateau_s"] = _plateau_band(out["ari_vs_valley"], PLATEAU_ARI)
    out["ari_vs_default"] = plateau_ari(per_ts, GAPS, ref_gap_s=DEFAULTS["gap_s"])
    return out


def build_spec(scope: str, patches: List[str], records: Sequence[MatchRecord], pairs: Dict[str, np.ndarray],
               n_boot: int = 200, seed: int = 7, bandwidth: float = 0.08,
               pooled: Optional[BoundarySpec] = None, radius_info: Optional[dict] = None) -> tuple:
    """Estimate every boundary for one slice; returns (spec, details)."""
    T = estimate_temporal(records, n_boot=n_boot, seed=seed, bandwidth=bandwidth)
    # --- G ---
    if T.get("ok"):
        gap, gap_src = float(T["kde"]["valley_s"]), "valley"
        b = T["bootstrap"].get("valley_s") or {}
        gap_ci = [b["p2.5"], b["p97.5"]] if b else None
        depth, plateau = float(T["kde"]["depth"]), T.get("plateau_s")
    elif pooled is not None:
        gap, gap_src, gap_ci, depth, plateau = pooled.gap_s, "pooled", pooled.gap_ci_s, None, pooled.gap_plateau_s
    else:
        gap, gap_src, gap_ci, depth, plateau = DEFAULTS["gap_s"], "default", None, None, None
    # --- D (window = this slice's G) ---
    S = estimate_spatial(pairs, gap_s=gap, n_boot=n_boot, seed=seed)
    if S.get("ok"):
        D, D_src = float(S["crossover_u"]), "crossover"
        bd = S.get("bootstrap") or {}
        D_ci = [bd["p2.5"], bd["p97.5"]] if bd else None
        D_mass = S.get("mass_near_boundary")
    elif pooled is not None:
        D, D_src, D_ci, D_mass = pooled.diameter_u, "pooled", pooled.diameter_ci_u, None
    else:
        D, D_src, D_ci, D_mass = DEFAULTS["diameter_u"], "default", None, None
    # --- R ---
    if radius_info and radius_info.get("ok"):
        R, R_src, R_cov = float(radius_info["radius_u"]), f"mechanics:{radius_info['radius_u']:.0f}u; coverage from datadragon:{radius_info['version']}", float(radius_info["coverage_at_reference"])
    elif pooled is not None and pooled.radius_source != "default":
        R, R_src, R_cov = pooled.validity_radius_u, pooled.radius_source, pooled.radius_coverage
    else:
        R, R_src, R_cov = DEFAULTS["validity_radius_u"], "default", None

    spec = BoundarySpec(
        scope=scope, patches=list(patches), n_matches=len(records), n_intervals=int(T.get("n_intervals", 0)),
        n_pairs_in_window=int(S.get("n_pairs_in_window", 0)),
        gap_s=gap, gap_source=gap_src, gap_ci_s=gap_ci, gap_depth=depth, gap_ok=bool(T.get("ok")),
        gap_plateau_s=plateau, plateau_ari=PLATEAU_ARI,
        diameter_u=D, diameter_source=D_src, diameter_ci_u=D_ci, diameter_mass_near=D_mass,
        validity_radius_u=R, radius_source=R_src, radius_coverage=R_cov,
        provenance={"created": _dt.datetime.now().isoformat(timespec="seconds"), "seed": seed, "n_boot": n_boot,
                    "bandwidth": bandwidth, "gaps_swept_s": GAPS},
    )
    return spec, {"temporal": T, "spatial": S, "radius": radius_info}


def drift_decision(per_patch: Dict[str, BoundarySpec], pooled: BoundarySpec,
                   diameter_rel_tol: float = DIAMETER_REL_TOL) -> dict:
    rows = {}
    all_inside = True
    for patch, s in per_patch.items():
        g_inside = (pooled.gap_plateau_s is not None and pooled.gap_plateau_s[0] <= s.gap_s <= pooled.gap_plateau_s[1]) \
            if s.gap_source == "valley" else None
        d_rel = abs(s.diameter_u - pooled.diameter_u) / pooled.diameter_u if s.diameter_source == "crossover" else None
        ci_overlap = None
        if s.diameter_ci_u and pooled.diameter_ci_u:
            ci_overlap = not (s.diameter_ci_u[1] < pooled.diameter_ci_u[0] or s.diameter_ci_u[0] > pooled.diameter_ci_u[1])
        d_inside = (d_rel is not None and d_rel <= diameter_rel_tol and (ci_overlap is None or ci_overlap)) \
            if d_rel is not None else None
        r_inside = None
        if s.radius_coverage is not None and pooled.radius_coverage is not None:
            r_inside = abs(s.radius_coverage - pooled.radius_coverage) <= RADIUS_COVERAGE_TOL
        inside = (g_inside is not False) and (d_inside is not False) and (r_inside is not False)
        all_inside &= inside
        rows[patch] = {"gap_s": s.gap_s, "gap_source": s.gap_source, "gap_inside_pooled_plateau": g_inside,
                       "diameter_u": s.diameter_u, "diameter_source": s.diameter_source, "diameter_rel_diff": d_rel,
                       "diameter_ci_overlap": ci_overlap, "diameter_inside": d_inside,
                       "radius_coverage": s.radius_coverage, "radius_inside": r_inside, "inside": inside}
    return {"verdict": "pooled" if all_inside else "per_patch", "pooled_gap_s": pooled.gap_s,
            "pooled_plateau_s": pooled.gap_plateau_s, "pooled_diameter_u": pooled.diameter_u,
            "diameter_rel_tol": diameter_rel_tol, "radius_coverage_tol": RADIUS_COVERAGE_TOL,
            "plateau_ari": PLATEAU_ARI, "patches": rows}


def drift_markdown(decision: dict, per_patch: Dict[str, BoundarySpec], pooled: BoundarySpec) -> str:
    lines = ["| scope | G (s) | 95% CI | depth | plateau | D (u) | 95% CI | R (u) | inside |", "|---|---|---|---|---|---|---|---|---|"]

    def row(name, s: BoundarySpec, inside):
        gci = f"{s.gap_ci_s[0]:.1f}-{s.gap_ci_s[1]:.1f}" if s.gap_ci_s else "-"
        dci = f"{s.diameter_ci_u[0]:.0f}-{s.diameter_ci_u[1]:.0f}" if s.diameter_ci_u else "-"
        pl = f"{s.gap_plateau_s[0]:g}-{s.gap_plateau_s[1]:g}" if s.gap_plateau_s else "-"
        dp = f"{s.gap_depth:.2f}" if s.gap_depth is not None else "-"
        return f"| {name} | {s.gap_s:.1f} ({s.gap_source}) | {gci} | {dp} | {pl} | {s.diameter_u:.0f} ({s.diameter_source}) | {dci} | {s.validity_radius_u:.0f} | {inside} |"

    lines.append(row("pooled", pooled, "-"))
    for p, s in per_patch.items():
        lines.append(row(p, s, decision["patches"][p]["inside"]))
    lines.append("")
    lines.append(f"Verdict: **{decision['verdict']}** (G inside pooled plateau at ARI >= {PLATEAU_ARI}; D within "
                 f"+-{int(DIAMETER_REL_TOL * 100)}% of pooled with overlapping CIs).")
    return "\n".join(lines)
