#!/usr/bin/env python3
"""RR3 close-out: publish match scope, timing/path parity checks, sparse s_Q fallback.

Does not refit V/q or retune matching tolerances after TEST. Reads RR3 artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

RR3 = REPO / "outputs" / "review_response_rr3_quiet_20260920"
BUNDLE = (
    REPO
    / "outputs"
    / "v_redesign_wave4_corrected_20260919"
    / "evaluators"
    / "A_MLP_expanded_evaluator.joblib"
)
LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
DT_MAX_MS = 120_000
DP_MAX = 0.025


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.{nd}f}"


def load_fights(lab_path: Path, D, L, roles):
    lab = np.load(lab_path, allow_pickle=False)
    E = D.load_engagements(L, "MAIN", roles, states=False, counts=False)
    e_key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(zip(E["match"].astype(str).tolist(), E["s"].astype(np.int64).tolist()))
    }
    idx_e, idx_l = [], []
    for i, (a, b) in enumerate(zip(lab["match"].astype(str).tolist(), lab["s"].astype(np.int64).tolist())):
        j = e_key.get((a, int(b)))
        if j is not None:
            idx_l.append(i)
            idx_e.append(j)
    idx_l = np.asarray(idx_l, int)
    idx_e = np.asarray(idx_e, int)
    ok = lab["Y_SVI"][idx_l] >= 0
    if "missing_score" in lab.files:
        ok &= lab["missing_score"][idx_l] == 0
    ok &= (E["pre_ok"][idx_e] == 1) & (E["valid_h90"][idx_e] == 1)
    idx_l, idx_e = idx_l[ok], idx_e[ok]
    s = E["s"][idx_e].astype(np.int64)
    endpoint = E["endpoint_h90"][idx_e].astype(np.int64)
    return dict(
        match=lab["match"][idx_l].astype(str),
        s=s,
        endpoint=endpoint,
        L=endpoint - s,
        p_pre=lab["p_pre"][idx_l].astype(float),
        delta_V=lab["delta_V"][idx_l].astype(float),
        B40=lab["B40"][idx_l].astype(bool) if "B40" in lab.files else (lab["p_pre"][idx_l] >= 0.4) & (lab["p_pre"][idx_l] <= 0.6),
        names=list(E["names"]),
    )


def coverage_by_bin(fights: Dict[str, Any], paired_keys: set) -> List[dict]:
    L_s = fights["L"].astype(float) / 1000.0
    t_min = fights["s"].astype(float) / 60000.0
    rows = []
    # L bins emphasizing the 120s hard constraint
    for lab, m in (
        ("L_le_60s", L_s <= 60),
        ("L_60_120s", (L_s > 60) & (L_s <= 120)),
        ("L_gt_120s", L_s > 120),
    ):
        keys = [(fights["match"][i], int(fights["s"][i])) for i in np.where(m)[0]]
        n = len(keys)
        n_m = sum(1 for k in keys if k in paired_keys)
        rows.append(dict(bin=lab, n=n, n_matched=n_m, coverage=float(n_m / n) if n else float("nan")))
    for lab, m in (
        ("t_2_10", (t_min >= 2) & (t_min < 10)),
        ("t_10_20", (t_min >= 10) & (t_min < 20)),
        ("t_20_30", (t_min >= 20) & (t_min < 30)),
        ("t_30_inf", t_min >= 30),
    ):
        keys = [(fights["match"][i], int(fights["s"][i])) for i in np.where(m)[0]]
        n = len(keys)
        n_m = sum(1 for k in keys if k in paired_keys)
        rows.append(dict(bin=lab, n=n, n_matched=n_m, coverage=float(n_m / n) if n else float("nan")))
    b40 = fights["B40"]
    for lab, m in (("B40", b40), ("outside_B40", ~b40)):
        keys = [(fights["match"][i], int(fights["s"][i])) for i in np.where(m)[0]]
        n = len(keys)
        n_m = sum(1 for k in keys if k in paired_keys)
        rows.append(dict(bin=lab, n=n, n_matched=n_m, coverage=float(n_m / n) if n else float("nan")))
    return rows


def rebuild_mu_s_with_fallback(pairs: Dict[str, np.ndarray], min_matches: int = 100) -> Dict[str, Any]:
    """Q_CAL quiet cells with declared fallback: L-bin → time-bin → p-bin → global."""
    from collections import defaultdict

    p = pairs["quiet_p_pre"]
    t = pairs["quiet_t0"].astype(float) / 60000.0
    L = pairs["quiet_L"].astype(float) / 1000.0
    d = pairs["quiet_delta_V"]
    g = pairs["match"].astype(str)
    p_edges = np.linspace(0.0, 1.0, 11)
    t_bands = [(2.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 1e9)]
    L_bands = [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 1e9)]

    def cell_stats(mask: np.ndarray) -> Tuple[float, float, int, int]:
        if not mask.any():
            return float("nan"), float("nan"), 0, 0
        _, inv, c = np.unique(g[mask], return_inverse=True, return_counts=True)
        w = 1.0 / c[inv]
        mu = float(np.average(d[mask], weights=w))
        s = float(np.quantile(np.abs(d[mask] - mu), 0.75))
        return mu, max(s, 1e-6), int(mask.sum()), int(len(np.unique(g[mask])))

    # precompute hierarchy
    global_mu, global_s, _, _ = cell_stats(np.ones(len(d), dtype=bool))
    cells = []
    n_fallback = defaultdict(int)
    for pi in range(10):
        for ti, (tlo, thi) in enumerate(t_bands):
            for Li, (Llo, Lhi) in enumerate(L_bands):
                m_full = (
                    (p >= p_edges[pi])
                    & (p < p_edges[pi + 1])
                    & (t >= tlo)
                    & (t < thi)
                    & (L >= Llo)
                    & (L < Lhi)
                )
                mu, s, n, n_m = cell_stats(m_full)
                level = "p_t_L"
                if n_m < min_matches:
                    # drop L
                    m2 = (p >= p_edges[pi]) & (p < p_edges[pi + 1]) & (t >= tlo) & (t < thi)
                    mu, s, n, n_m = cell_stats(m2)
                    level = "p_t"
                    n_fallback["to_p_t"] += 1
                if n_m < min_matches:
                    m3 = (p >= p_edges[pi]) & (p < p_edges[pi + 1])
                    mu, s, n, n_m = cell_stats(m3)
                    level = "p"
                    n_fallback["to_p"] += 1
                if n_m < min_matches:
                    mu, s, n, n_m = global_mu, global_s, int(len(d)), int(len(np.unique(g)))
                    level = "global"
                    n_fallback["to_global"] += 1
                cells.append(
                    dict(
                        p_bin=pi,
                        t_bin=ti,
                        L_bin=Li,
                        n=n,
                        n_matches=n_m,
                        mu=mu,
                        s=s,
                        level=level,
                        supported=(level != "global" or n_m >= min_matches),
                    )
                )
    return dict(
        min_matches=min_matches,
        fallback_order=["p_t_L", "p_t", "p", "global"],
        fallback_counts=dict(n_fallback),
        p_edges=p_edges.tolist(),
        t_bands=t_bands,
        L_bands=L_bands,
        global_mu=global_mu,
        global_s=global_s,
        cells=cells,
        note=(
            "DEV(Q_CAL) quiet only. Sparse cells fall back L→time→p→global. "
            "L>120s fights are out of RR3 match support; do not treat their s_Q as well identified."
        ),
    )


def path_parity_sample(
    pairs_path: Path,
    fights: Dict[str, Any],
    n_sample: int,
    seed: int,
) -> Dict[str, Any]:
    """Re-score fight and quiet via StateBuilder; compare stored fight labels vs rebuild."""
    data_root = _data_root()
    _setup(data_root)
    import fc20260915_common as C
    import data.cache_io as cio
    from core.config import NODE_FEATURE_NAMES
    from gameplay.state_value_v2 import StateBuilder, state_matrix
    from gameplay.state_value import final_outcome
    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    ev = load_evaluator(BUNDLE)
    z = np.load(pairs_path, allow_pickle=False)
    n = len(z["match"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)

    # index fights
    fkey = {(a, int(b)): i for i, (a, b) in enumerate(zip(fights["match"].tolist(), fights["s"].tolist()))}
    names = fights["names"]

    abs_err_p = []
    abs_err_d = []
    meas_L_fight = []
    meas_L_quiet_inclusive = []  # t0 .. t0+L-1  => span L-1
    meas_L_quiet_endpoint = []  # if scored at t0+L
    n_ok = n_fail = 0

    for j in idx:
        mid = str(z["match"][j])
        s = int(z["fight_s"][j])
        L = int(z["fight_L"][j])
        tq = int(z["quiet_t0"][j])
        fi = fkey.get((mid, s))
        if fi is None:
            n_fail += 1
            continue
        pack = cio.load_match_cache(mid)
        if pack is None:
            n_fail += 1
            continue
        try:
            final_outcome(pack["events"])
            builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
            # Fight: score at s and endpoint-1 (stored L = endpoint-s)
            endpoint = int(fights["endpoint"][fi])
            t_post_f = endpoint - 1 if endpoint > s else endpoint
            Xf = state_matrix([builder.at(s), builder.at(t_post_f)], names)
            pf = predict_calibrated(ev, Xf)
            # Quiet as shipped: t0 and t0+L-1
            t_post_q = tq + L - 1 if L > 0 else tq
            Xq = state_matrix([builder.at(tq), builder.at(t_post_q)], names)
            pq = predict_calibrated(ev, Xq)
            # Alternate quiet end at t0+L (exclusive-style end query)
            try:
                Xq2 = state_matrix([builder.at(tq), builder.at(tq + L)], names)
                pq2 = predict_calibrated(ev, Xq2)
                d_alt = float(pq2[1] - pq2[0])
            except Exception:
                d_alt = float("nan")
        except Exception:
            n_fail += 1
            continue
        n_ok += 1
        p_store = float(z["fight_p_pre"][j])
        d_store = float(z["fight_delta_V"][j])
        abs_err_p.append(abs(float(pf[0]) - p_store))
        abs_err_d.append(abs(float(pf[1] - pf[0]) - d_store))
        meas_L_fight.append(t_post_f - s)
        meas_L_quiet_inclusive.append(t_post_q - tq)
        meas_L_quiet_endpoint.append(L)
        # quiet stored delta vs rebuild
        # (also track quiet rebuild error vs stored quiet delta)
        # stored in pairs
        # append into lists via side channel
        if not hasattr(path_parity_sample, "_qd"):
            path_parity_sample._qd = []
            path_parity_sample._qd_alt = []
        path_parity_sample._qd.append(abs(float(pq[1] - pq[0]) - float(z["quiet_delta_V"][j])))
        path_parity_sample._qd_alt.append(abs(d_alt - float(z["quiet_delta_V"][j])) if not math.isnan(d_alt) else float("nan"))

    qd_err = getattr(path_parity_sample, "_qd", [])
    qd_alt = getattr(path_parity_sample, "_qd_alt", [])
    # clear side state
    if hasattr(path_parity_sample, "_qd"):
        del path_parity_sample._qd
        del path_parity_sample._qd_alt

    def summ(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return dict(n=0, mean=float("nan"), p50=float("nan"), p95=float("nan"), max=float("nan"))
        return dict(n=int(len(a)), mean=float(a.mean()), p50=float(np.median(a)), p95=float(np.quantile(a, 0.95)), max=float(a.max()))

    return dict(
        n_requested=int(len(idx)),
        n_ok=n_ok,
        n_fail=n_fail,
        fight_rebuild_abs_err_p_pre=summ(abs_err_p),
        fight_rebuild_abs_err_delta=summ(abs_err_d),
        quiet_rebuild_abs_err_delta_Lminus1=summ(qd_err),
        quiet_rebuild_abs_err_delta_if_end_t0_plus_L=summ(qd_alt),
        measured_span_ms=dict(
            fight_post_minus_pre_mean=float(np.mean(meas_L_fight)) if meas_L_fight else float("nan"),
            quiet_shipped_t0_to_t0_plus_L_minus_1_mean=float(np.mean(meas_L_quiet_inclusive)) if meas_L_quiet_inclusive else float("nan"),
            stored_L_mean=float(np.mean(meas_L_quiet_endpoint)) if meas_L_quiet_endpoint else float("nan"),
            note=(
                "Shipped quiet scores at [t0, t0+L-1] (span L-1 ms). "
                "Stored L=endpoint-s. Fight rebuild uses [s, endpoint-1]. "
                "1 ms span difference is documented; not assumed to explain |ΔV| gap."
            ),
        ),
        convention_lock=(
            "Going forward report measurement length as (post_query_ms - pre_query_ms). "
            "Shipped RR3 quiet used post=t0+L-1; overlap/kill checks used [t0, t0+L]."
        ),
    )


def main() -> int:
    contrast = json.loads((RR3 / "quiet_contrast.json").read_text(encoding="utf-8"))
    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    Lyt = D.Layout(False)
    print("load fights for coverage bins…", flush=True)
    TE = load_fights(LAB / "TEST_h90.npz", D, Lyt, ["TEST"])
    CA = load_fights(LAB / "Q_CAL_h90.npz", D, Lyt, ["Q_CAL"])

    te_pairs = np.load(RR3 / "quiet_pairs_TEST.npz", allow_pickle=False)
    ca_pairs = np.load(RR3 / "quiet_pairs_Q_CAL.npz", allow_pickle=False)
    te_keys = set(zip(te_pairs["match"].astype(str).tolist(), te_pairs["fight_s"].astype(np.int64).tolist()))
    ca_keys = set(zip(ca_pairs["match"].astype(str).tolist(), ca_pairs["fight_s"].astype(np.int64).tolist()))

    # logical constraint
    L_te = TE["L"].astype(float)
    impossible = float(np.mean(L_te > DT_MAX_MS))
    scope = dict(
        matching_tolerances=dict(dp_max=DP_MAX, dt_max_ms=DT_MAX_MS),
        logical_constraint=(
            "With equal length L and no overlap with [s,s+L], any quiet start must satisfy "
            "|t-s|>=L. Combined with |t-s|<=120s, fights with L>120s are unmatchable under this rule."
        ),
        TEST_share_L_gt_120s=impossible,
        TEST_n_L_gt_120s=int(np.sum(L_te > DT_MAX_MS)),
        TEST_coverage_bins=coverage_by_bin(TE, te_keys),
        Q_CAL_coverage_bins=coverage_by_bin(CA, ca_keys),
        TEST_matched_traits=contrast["TEST"].get("matched_fight_traits"),
        TEST_unmatched_traits=contrast["TEST"].get("unmatched_fight_traits"),
        Q_CAL_matched_traits=contrast["Q_CAL"].get("matched_fight_traits"),
        Q_CAL_unmatched_traits=contrast["Q_CAL"].get("unmatched_fight_traits"),
        summarize_delta_note=(
            "In RR3 tables, means / mean|ΔV| / P(ΔV>0) are match-weighted; "
            "median and quantiles are row-level (unweighted). Do not label the whole table as match-weighted."
        ),
    )

    print("rebuild mu/s_Q with sparse fallback…", flush=True)
    mu_s = rebuild_mu_s_with_fallback({k: ca_pairs[k] for k in ca_pairs.files})
    (RR3 / "quiet_mu_s_Q_CAL_fallback.json").write_text(json.dumps(scrub(mu_s), indent=2) + "\n", encoding="utf-8")

    print("path/timing parity sample (n=200)…", flush=True)
    parity = path_parity_sample(RR3 / "quiet_pairs_TEST.npz", TE, n_sample=200, seed=7)

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic="EXPLORATORY_RR3_CLOSEOUT_PRIOR_TEST_EXPOSURE",
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §7",
        parent_commit_note="Closes RR3 reporting gaps without retuning match tolerances or refitting V/q",
        scope=scope,
        path_parity=parity,
        mu_s_fallback=dict(
            path="outputs/review_response_rr3_quiet_20260920/quiet_mu_s_Q_CAL_fallback.json",
            fallback_counts=mu_s["fallback_counts"],
            min_matches=mu_s["min_matches"],
        ),
        preserved_primary_contrast=contrast["TEST"].get("contrast"),
    )
    (RR3 / "rr3_closeout.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")
    (REPO / "docs" / "REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.json").write_text(
        json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8"
    )

    # markdown
    lines = [
        "# RR3 close-out — match scope, timing parity, sparse \(s_Q\)",
        "",
        f"Generated: {payload['generated']}",
        "Parent results: [REVIEW_RESPONSE_RR3_QUIET_20260920.md](REVIEW_RESPONSE_RR3_QUIET_20260920.md)",
        "",
        "## Scope lock (do not over-claim)",
        "",
        "> On the **matchable subset** under same-match / |Δp|≤0.025 / |Δt|≤120s / equal \(L_i\), fight |ΔV| exceeds quiet |ΔV|. "
        "Not: all fights; not causal fight-only contribution; not q predicting pure fight effect.",
        "",
        "### Why coverage ≈28%",
        "",
        f"- Logical: if quiet must not overlap \([s,s+L]\) and \(|t-s|\\le 120\)s with equal length, then **\(L>120\)s is unmatchable**.",
        f"- TEST share with \(L>120\)s: **{fmt(100*impossible, 1)}%** (n={scope['TEST_n_L_gt_120s']}).",
        "- Empirically, unmatched fights are longer on average (see traits).",
        "",
        "## Matched vs unmatched traits (TEST)",
        "",
        "| Group | n | mean p_pre | mean t (min) | mean L (s) | mean |ΔV| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lab, tr in (
        ("Matched", scope["TEST_matched_traits"]),
        ("Unmatched", scope["TEST_unmatched_traits"]),
    ):
        lines.append(
            f"| {lab} | {tr['n']} | {fmt(tr['mean_p'])} | {fmt(tr['mean_t_min'])} | {fmt(tr['mean_L_s'])} | {fmt(tr['mean_abs_dV'])} |"
        )
    lines += [
        "",
        "## Coverage by length / time / B40 (TEST)",
        "",
        "| Bin | n | matched | coverage |",
        "|---|---:|---:|---:|",
    ]
    for r in scope["TEST_coverage_bins"]:
        lines.append(f"| {r['bin']} | {r['n']} | {r['n_matched']} | {fmt(r['coverage'], 3)} |")

    pe = parity["fight_rebuild_abs_err_p_pre"]
    de = parity["fight_rebuild_abs_err_delta"]
    qe = parity["quiet_rebuild_abs_err_delta_Lminus1"]
    lines += [
        "",
        "## Timing / state-path parity (TEST sample)",
        "",
        f"- Sample ok={parity['n_ok']} fail={parity['n_fail']} (seed=7).",
        f"- Fight rebuild |Δp_pre| vs stored label: mean={fmt(pe['mean'], 5)} p95={fmt(pe['p95'], 5)}",
        f"- Fight rebuild |Δ(ΔV)| vs stored label: mean={fmt(de['mean'], 5)} p95={fmt(de['p95'], 5)}",
        f"- Quiet rebuild |Δ(ΔV)| (shipped \(t_0+L-1\)): mean={fmt(qe['mean'], 5)} p95={fmt(qe['p95'], 5)}",
        f"- Measured spans: fight mean={fmt(parity['measured_span_ms']['fight_post_minus_pre_mean'], 1)} ms; "
        f"quiet shipped mean={fmt(parity['measured_span_ms']['quiet_shipped_t0_to_t0_plus_L_minus_1_mean'], 1)} ms; "
        f"stored L mean={fmt(parity['measured_span_ms']['stored_L_mean'], 1)} ms.",
        "",
        "**Convention note:** shipped quiet scored \([t_0,\\,t_0+L-1]\) while overlap/kill used \([t_0,\\,t_0+L]\). "
        "Report measurement length as post−pre query times. The ~1 ms span gap is recorded; it is not treated as the explanation of the ~6%p |ΔV| contrast.",
        "",
        "## Statistics labeling",
        "",
        scope["summarize_delta_note"],
        "",
        "## \(s_Q\) sparse-cell rule (for RR4)",
        "",
        f"- Fallback order: `{mu_s['fallback_order']}` with min_matches={mu_s['min_matches']}.",
        f"- Fallback counts: `{json.dumps(mu_s['fallback_counts'])}`.",
        "- Artifact: `outputs/review_response_rr3_quiet_20260920/quiet_mu_s_Q_CAL_fallback.json`",
        "- Do not impute unsupported long-\(L\) regions with ε defaults for ‘small-ΔV’ filters.",
        "",
        "## Primary RR3 contrast (unchanged)",
        "",
        f"- TEST E[|ΔV|_fight − |ΔV|_quiet]={fmt(payload['preserved_primary_contrast']['mean_abs_fight_minus_quiet']['estimate'], 5)} "
        f"CI95={payload['preserved_primary_contrast']['mean_abs_fight_minus_quiet']['ci95']}",
        "",
    ]
    md = REPO / "docs" / "REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # patch main RR3 md with scope pointer
    main_md = REPO / "docs" / "REVIEW_RESPONSE_RR3_QUIET_20260920.md"
    text = main_md.read_text(encoding="utf-8")
    if "RR3_CLOSEOUT" not in text:
        text = text.replace(
            "## Guardrails",
            "Close-out (traits, \(L>120\)s constraint, timing parity, \(s_Q\) fallback): "
            "[REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md](REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md)\n\n## Guardrails",
        )
        main_md.write_text(text, encoding="utf-8")

    print("wrote", md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
