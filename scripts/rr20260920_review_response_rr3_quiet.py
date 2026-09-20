#!/usr/bin/env python3
"""RR3 — new-V (fit85) matched no-kill quiet reference vs teamfight ΔV.

Design: docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §7
Locks: frozen fit85 MLP bundle; actual L_i = endpoint_h90 - s; same-match 1:1;
       |Δp|≤0.025, |Δt|≤120s (proposal values); objects/farming allowed.
Does NOT reuse old_V quiet ratios. Does NOT change q inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

BUNDLE = (
    REPO
    / "outputs"
    / "v_redesign_wave4_corrected_20260919"
    / "evaluators"
    / "A_MLP_expanded_evaluator.joblib"
)
LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
OUT = REPO / "outputs" / "review_response_rr3_quiet_20260920"
ROLE = "EXPLORATORY_RR3_NEWV_QUIET_PRIOR_TEST_EXPOSURE"

# Design proposal values (locked for this run after DEV coverage check)
DP_MAX = 0.025
DT_MAX_MS = 120_000
STEP_MS = 15_000
START_MIN_MS = 120_000


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    # worktree FIRST so gameplay.state_value_v2 resolves (repo has shadow gameplay/)
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.{nd}f}"


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


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def tie_hash(*parts) -> int:
    h = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def kill_times(events: List[dict]) -> np.ndarray:
    return np.asarray(
        [int(e["timestamp"]) for e in events if e.get("type") == "CHAMPION_KILL"],
        dtype=np.int64,
    )


def no_kill(kills: np.ndarray, t0: int, t1: int) -> bool:
    if kills.size == 0:
        return True
    return not bool(np.any((kills >= t0) & (kills <= t1)))


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def load_fight_pack(lab_path: Path, D, L, roles: List[str]) -> Dict[str, Any]:
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
    ok &= E["pre_ok"][idx_e] == 1
    ok &= E["valid_h90"][idx_e] == 1
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
        Y=lab["Y_SVI"][idx_l].astype(np.int8),
        B40=lab["B40"][idx_l].astype(bool) if "B40" in lab.files else None,
        names=list(E["names"]),
    )


def summarize_delta(delta: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    if len(delta) == 0:
        return dict(n=0, n_matches=0, mean=float("nan"), median=float("nan"),
                    mean_abs=float("nan"), median_abs=float("nan"), p_pos=float("nan"),
                    q25_abs=float("nan"), q75_abs=float("nan"), q90_abs=float("nan"))
    w = match_weights(g)
    absd = np.abs(delta)
    return dict(
        n=int(len(delta)),
        n_matches=int(len(np.unique(g.astype(str)))),
        mean=float(np.average(delta, weights=w)),
        median=float(np.median(delta)),
        mean_abs=float(np.average(absd, weights=w)),
        median_abs=float(np.median(absd)),
        p_pos=float(np.average((delta > 0).astype(float), weights=w)),
        q25_abs=float(np.quantile(absd, 0.25)),
        q75_abs=float(np.quantile(absd, 0.75)),
        q90_abs=float(np.quantile(absd, 0.90)),
    )


def bootstrap_paired_diff(fight_d, quiet_d, g, reps=2000, seed=7):
    """Match-clustered bootstrap of mean(fight|ΔV|) - mean(quiet|ΔV|) and signed means."""
    g = np.asarray(g).astype(str)
    fd = np.asarray(fight_d, float)
    qd = np.asarray(quiet_d, float)
    matches, inv = np.unique(g, return_inverse=True)
    n_m = len(matches)
    # per-match means of abs and signed
    sum_fa = np.zeros(n_m)
    sum_qa = np.zeros(n_m)
    sum_fs = np.zeros(n_m)
    sum_qs = np.zeros(n_m)
    cnt = np.zeros(n_m)
    np.add.at(sum_fa, inv, np.abs(fd))
    np.add.at(sum_qa, inv, np.abs(qd))
    np.add.at(sum_fs, inv, fd)
    np.add.at(sum_qs, inv, qd)
    np.add.at(cnt, inv, 1.0)
    m_fa = sum_fa / cnt
    m_qa = sum_qa / cnt
    m_fs = sum_fs / cnt
    m_qs = sum_qs / cnt
    obs_abs = float(np.mean(m_fa - m_qa))
    obs_signed = float(np.mean(m_fs - m_qs))
    obs_ppos = float(np.mean((fd > 0).astype(float) - (qd > 0).astype(float)))
    # ppos also match-averaged
    sum_fp = np.zeros(n_m)
    sum_qp = np.zeros(n_m)
    np.add.at(sum_fp, inv, (fd > 0).astype(float))
    np.add.at(sum_qp, inv, (qd > 0).astype(float))
    m_fp = sum_fp / cnt
    m_qp = sum_qp / cnt
    obs_ppos = float(np.mean(m_fp - m_qp))
    rng = np.random.default_rng(seed)
    abs_draws, signed_draws, ppos_draws = [], [], []
    for _ in range(reps):
        samp = rng.integers(0, n_m, size=n_m)
        abs_draws.append(float(np.mean(m_fa[samp] - m_qa[samp])))
        signed_draws.append(float(np.mean(m_fs[samp] - m_qs[samp])))
        ppos_draws.append(float(np.mean(m_fp[samp] - m_qp[samp])))
    def pack(obs, draws):
        draws = np.asarray(draws)
        return dict(
            estimate=obs,
            ci95=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            bootstrap_fraction_positive=float(np.mean(draws > 0)),
        )
    return dict(
        mean_abs_fight_minus_quiet=pack(obs_abs, abs_draws),
        mean_signed_fight_minus_quiet=pack(obs_signed, signed_draws),
        p_pos_fight_minus_quiet=pack(obs_ppos, ppos_draws),
    )


def process_role(
    name: str,
    fights: Dict[str, Any],
    ev,
    load_match_cache,
    StateBuilder,
    state_matrix,
    final_outcome,
    NODE_FEATURE_NAMES,
    predict_calibrated,
    max_matches: Optional[int] = None,
) -> Dict[str, Any]:
    names = fights["names"]
    by_match: Dict[str, List[int]] = defaultdict(list)
    for i, mid in enumerate(fights["match"].tolist()):
        by_match[mid].append(i)

    match_ids = sorted(by_match.keys())
    if max_matches is not None:
        match_ids = match_ids[: int(max_matches)]

    paired_fight_idx: List[int] = []
    quiet_p: List[float] = []
    quiet_d: List[float] = []
    quiet_t0: List[int] = []
    quiet_L: List[int] = []
    unmatched_reason: Dict[str, int] = defaultdict(int)
    n_cache_miss = n_builder_fail = n_loaded = 0
    used_quiet_intervals: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    t_wall0 = time.time()
    for mi, mid in enumerate(match_ids):
        if mi % 500 == 0:
            print(f"  [{name}] match {mi}/{len(match_ids)} paired={len(paired_fight_idx)}", flush=True)
        pack = load_match_cache(mid)
        if pack is None:
            n_cache_miss += 1
            for _ in by_match[mid]:
                unmatched_reason["cache_miss"] += 1
            continue
        try:
            _w, terminal = final_outcome(pack["events"])
            builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
            kills = kill_times(pack["events"])
        except Exception:
            n_builder_fail += 1
            for _ in by_match[mid]:
                unmatched_reason["builder_fail"] += 1
            continue
        n_loaded += 1
        last = int(terminal if terminal and terminal > 0 else pack["minute_ts"][-1])
        last = int(min(last, int(pack["minute_ts"][-1])))

        fight_ix = by_match[mid]
        fight_intervals = [(int(fights["s"][i]), int(fights["endpoint"][i])) for i in fight_ix]

        # score cache for (t0, t1) endpoints
        score_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}

        def score_window(t0: int, L: int) -> Optional[Tuple[float, float]]:
            t1 = t0 + L - 1 if L > 0 else t0
            key = (t0, t1)
            if key in score_cache:
                return score_cache[key]
            try:
                X = state_matrix([builder.at(t0), builder.at(t1)], names)
                p = predict_calibrated(ev, X)
                out = (float(p[0]), float(p[1] - p[0]))
            except Exception:
                return None
            score_cache[key] = out
            return out

        # greedy: process fights in time order
        for i in sorted(fight_ix, key=lambda j: int(fights["s"][j])):
            s_i = int(fights["s"][i])
            L_i = int(fights["L"][i])
            p_i = float(fights["p_pre"][i])
            if L_i <= 0:
                unmatched_reason["bad_L"] += 1
                continue
            t_hi = last - L_i
            if t_hi < START_MIN_MS:
                unmatched_reason["no_time_room"] += 1
                continue
            # candidate starts only near the fight (design |Δt|≤DT_MAX)
            t_lo = max(START_MIN_MS, s_i - DT_MAX_MS)
            t_hi_local = min(t_hi, s_i + DT_MAX_MS)
            if t_hi_local < t_lo:
                unmatched_reason["no_time_room"] += 1
                continue
            best = None  # (dist, tie, t0, p_q, d_q)
            for t0 in range(t_lo, t_hi_local + 1, STEP_MS):
                t1 = t0 + L_i
                # overlap fight intervals (including this fight)
                if any(overlaps(t0, t1, a, b) for a, b in fight_intervals):
                    continue
                # overlap already used quiet
                if any(overlaps(t0, t1, a, b) for a, b in used_quiet_intervals[mid]):
                    continue
                if not no_kill(kills, t0, t1):
                    continue
                scored = score_window(t0, L_i)
                if scored is None:
                    continue
                pq, dq = scored
                if abs(pq - p_i) > DP_MAX:
                    continue
                dist = (abs(pq - p_i) / DP_MAX) ** 2 + (abs(t0 - s_i) / DT_MAX_MS) ** 2
                th = tie_hash(mid, s_i, t0, L_i)
                cand = (dist, th, t0, pq, dq)
                if best is None or cand < best:
                    best = cand
            if best is None:
                unmatched_reason["no_candidate"] += 1
                continue
            _dist, _th, t0, pq, dq = best
            used_quiet_intervals[mid].append((t0, t0 + L_i))
            paired_fight_idx.append(i)
            quiet_p.append(pq)
            quiet_d.append(dq)
            quiet_t0.append(t0)
            quiet_L.append(L_i)

    paired_fight_idx = np.asarray(paired_fight_idx, int)
    n_fight = len(fights["match"])
    coverage = float(len(paired_fight_idx) / n_fight) if n_fight else 0.0

    if len(paired_fight_idx) == 0:
        empty_pairs = dict(
            match=np.zeros(0, dtype="U1"),
            fight_s=np.zeros(0, np.int64),
            fight_L=np.zeros(0, np.int64),
            fight_p_pre=np.zeros(0, float),
            fight_delta_V=np.zeros(0, float),
            quiet_t0=np.zeros(0, np.int64),
            quiet_L=np.zeros(0, np.int64),
            quiet_p_pre=np.zeros(0, float),
            quiet_delta_V=np.zeros(0, float),
            Y=np.zeros(0, np.int8),
        )
        return dict(
            role=name,
            n_fight=n_fight,
            n_matched=0,
            coverage=0.0,
            n_matches_requested=len(by_match),
            n_matches_processed=len(match_ids),
            n_cache_miss=n_cache_miss,
            n_builder_fail=n_builder_fail,
            n_loaded=n_loaded,
            unmatched_reason=dict(unmatched_reason),
            wall_s=time.time() - t_wall0,
        ), empty_pairs

    fd = fights["delta_V"][paired_fight_idx]
    fp = fights["p_pre"][paired_fight_idx]
    ft = fights["s"][paired_fight_idx].astype(float) / 60000.0
    fL = fights["L"][paired_fight_idx].astype(float)
    fg = fights["match"][paired_fight_idx]
    qd = np.asarray(quiet_d, float)
    qp = np.asarray(quiet_p, float)
    qt0 = np.asarray(quiet_t0, np.int64)
    qL = np.asarray(quiet_L, np.int64)

    # balance
    bal = dict(
        mean_abs_dp=float(np.mean(np.abs(fp - qp))),
        mean_abs_dt_s=float(np.mean(np.abs(fights["s"][paired_fight_idx] - qt0)) / 1000.0),
        mean_abs_dL_ms=float(np.mean(np.abs(fL - qL.astype(float)))),
        max_abs_dp=float(np.max(np.abs(fp - qp))),
        max_abs_dt_s=float(np.max(np.abs(fights["s"][paired_fight_idx] - qt0)) / 1000.0),
    )

    # unmatched fight traits
    matched_set = set(paired_fight_idx.tolist())
    um = np.array([i not in matched_set for i in range(n_fight)], dtype=bool)
    def traits(mask):
        if not mask.any():
            return {}
        return dict(
            n=int(mask.sum()),
            mean_p=float(np.mean(fights["p_pre"][mask])),
            mean_t_min=float(np.mean(fights["s"][mask]) / 60000.0),
            mean_L_s=float(np.mean(fights["L"][mask]) / 1000.0),
            mean_abs_dV=float(np.mean(np.abs(fights["delta_V"][mask]))),
        )

    contrast = bootstrap_paired_diff(fd, qd, fg)
    out = dict(
        role=name,
        n_fight=n_fight,
        n_matched=int(len(paired_fight_idx)),
        coverage=coverage,
        n_matches_requested=len(by_match),
        n_matches_processed=len(match_ids),
        n_cache_miss=n_cache_miss,
        n_builder_fail=n_builder_fail,
        n_loaded=n_loaded,
        unmatched_reason=dict(unmatched_reason),
        matching=dict(dp_max=DP_MAX, dt_max_ms=DT_MAX_MS, step_ms=STEP_MS, start_min_ms=START_MIN_MS),
        balance=bal,
        matched_fight=summarize_delta(fd, fg),
        matched_quiet=summarize_delta(qd, fg),
        unmatched_fight_traits=traits(um),
        matched_fight_traits=traits(~um),
        contrast=contrast,
        wall_s=time.time() - t_wall0,
    )
    # store pairs for downstream RR4
    pairs = dict(
        match=fg,
        fight_s=fights["s"][paired_fight_idx],
        fight_L=fights["L"][paired_fight_idx],
        fight_p_pre=fp,
        fight_delta_V=fd,
        quiet_t0=qt0,
        quiet_L=qL,
        quiet_p_pre=qp,
        quiet_delta_V=qd,
        Y=fights["Y"][paired_fight_idx],
    )
    return out, pairs


def estimate_mu_s_from_quiet(pairs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """DEV-only reference cells: p(.1) × time(4) × duration bins; mean ΔV and |ΔV-μ| q75."""
    p = pairs["quiet_p_pre"]
    t = pairs["quiet_t0"].astype(float) / 60000.0
    L = pairs["quiet_L"].astype(float) / 1000.0
    d = pairs["quiet_delta_V"]
    g = pairs["match"]
    p_edges = np.linspace(0.0, 1.0, 11)
    t_bands = [(2.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 1e9)]
    L_bands = [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 1e9)]
    cells = []
    for pi in range(10):
        for ti, (tlo, thi) in enumerate(t_bands):
            for Li, (Llo, Lhi) in enumerate(L_bands):
                m = (p >= p_edges[pi]) & (p < p_edges[pi + 1]) & (t >= tlo) & (t < thi) & (L >= Llo) & (L < Lhi)
                n_m = int(len(np.unique(g[m]))) if m.any() else 0
                if n_m < 100:
                    # fallback will be applied at query time; still record
                    pass
                if not m.any():
                    cells.append(dict(p_bin=pi, t_bin=ti, L_bin=Li, n=0, n_matches=0, mu=float("nan"), s=float("nan")))
                    continue
                w = match_weights(g[m])
                mu = float(np.average(d[m], weights=w))
                centered = np.abs(d[m] - mu)
                s = float(np.quantile(centered, 0.75))
                cells.append(dict(p_bin=pi, t_bin=ti, L_bin=Li, n=int(m.sum()), n_matches=n_m, mu=mu, s=max(s, 1e-6)))
    return dict(
        p_edges=p_edges.tolist(),
        t_bands=t_bands,
        L_bands=L_bands,
        cells=cells,
        note="Estimated on Q_CAL quiet only; epsilon=1e-6; not an OOS drift oracle for TEST.",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-reps", type=int, default=2000)
    ap.add_argument("--max-matches", type=int, default=None, help="debug cap per role")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args(argv)

    if not BUNDLE.is_file():
        raise SystemExit(f"missing bundle {BUNDLE}")
    for need in (LAB / "Q_CAL_h90.npz", LAB / "TEST_h90.npz"):
        if not need.is_file():
            raise SystemExit(f"missing {need}")

    data_root = _data_root()
    _setup(data_root)

    import fc20260915_common as C
    import fc20260915_data as D
    import data.cache_io as cio
    from core.config import NODE_FEATURE_NAMES
    from gameplay.state_value_v2 import StateBuilder, state_matrix
    from gameplay.state_value import final_outcome
    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    # Point at MAIN match packs on D: (declared in fc20260915_common.CACHE_MAIN)
    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    print(f"CACHE_DIR={cio.CACHE_DIR} exists={cio.CACHE_DIR.is_dir()}", flush=True)

    L = D.Layout(False)
    ev = load_evaluator(BUNDLE)
    v_sha = sha16(BUNDLE)

    print("load Q_CAL fights…", flush=True)
    CA = load_fight_pack(LAB / "Q_CAL_h90.npz", D, L, ["Q_CAL"])
    print(f"  Q_CAL n={len(CA['match'])}", flush=True)

    print("RR3 match quiet on Q_CAL (DEV)…", flush=True)
    ca_out, ca_pairs = process_role(
        "Q_CAL",
        CA,
        ev,
        cio.load_match_cache,
        StateBuilder,
        state_matrix,
        final_outcome,
        NODE_FEATURE_NAMES,
        predict_calibrated,
        max_matches=args.max_matches,
    )
    print(
        f"  Q_CAL matched={ca_out.get('n_matched')} coverage={fmt(ca_out.get('coverage'), 3)} "
        f"cache_miss={ca_out.get('n_cache_miss')}",
        flush=True,
    )

    mu_s = None
    if ca_out.get("n_matched", 0) > 50:
        mu_s = estimate_mu_s_from_quiet(ca_pairs)

    te_out = None
    te_pairs = None
    if not args.skip_test:
        print("load TEST fights…", flush=True)
        TE = load_fight_pack(LAB / "TEST_h90.npz", D, L, ["TEST"])
        print(f"  TEST n={len(TE['match'])}", flush=True)
        print("RR3 match quiet on TEST…", flush=True)
        te_out, te_pairs = process_role(
            "TEST",
            TE,
            ev,
            cio.load_match_cache,
            StateBuilder,
            state_matrix,
            final_outcome,
            NODE_FEATURE_NAMES,
            predict_calibrated,
            max_matches=args.max_matches,
        )
        # recompute contrast with requested boot reps
        if te_out.get("n_matched", 0) > 0:
            te_out["contrast"] = bootstrap_paired_diff(
                te_pairs["fight_delta_V"], te_pairs["quiet_delta_V"], te_pairs["match"], args.boot_reps
            )
        print(
            f"  TEST matched={te_out.get('n_matched')} coverage={fmt(te_out.get('coverage'), 3)} "
            f"cache_miss={te_out.get('n_cache_miss')}",
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    if ca_pairs is not None and ca_out.get("n_matched", 0) > 0:
        np.savez_compressed(OUT / "quiet_pairs_Q_CAL.npz", **ca_pairs)
    if te_pairs is not None and te_out and te_out.get("n_matched", 0) > 0:
        np.savez_compressed(OUT / "quiet_pairs_TEST.npz", **te_pairs)
    if mu_s is not None:
        (OUT / "quiet_mu_s_Q_CAL.json").write_text(json.dumps(scrub(mu_s), indent=2) + "\n", encoding="utf-8")

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §7",
        V_bundle=str(BUNDLE.relative_to(REPO)).replace("\\", "/"),
        V_sha16=v_sha,
        CACHE_DIR=str(cio.CACHE_DIR),
        matching=dict(dp_max=DP_MAX, dt_max_ms=DT_MAX_MS, step_ms=STEP_MS, start_min_ms=START_MIN_MS, note="proposal values"),
        Q_CAL=ca_out,
        TEST=te_out,
        interpretation=(
            "Quiet is a same-match no-kill reference of equal observed length L_i under frozen fit85. "
            "Not a causal fight effect. Objects/farming/time allowed. "
            "L_i is post information used only for matching/measurement."
        ),
    )
    (OUT / "quiet_contrast.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    # markdown
    lines = [
        "# RR3 — new-V matched quiet reference (fit85 MLP)",
        "",
        f"Generated: {payload['generated']}",
        f"**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md) §7",
        f"**V bundle sha16:** `{v_sha}` · **CACHE:** `{cio.CACHE_DIR}`",
        f"**Matching:** |Δp|≤{DP_MAX}, |Δt|≤{DT_MAX_MS/1000:.0f}s, step={STEP_MS/1000:.0f}s, actual L_i (not constant 90s)",
        "",
        "## Claim scope",
        "",
        "> Under frozen fit85, matched no-kill windows of the same observed length show how large engagement ΔV is versus background drift. Not a causal fight effect; not old_V quiet reuse.",
        "",
    ]

    def block(title: str, o: Optional[dict]):
        if not o:
            return [f"## {title}", "", "Not run.", ""]
        mf, mq = o.get("matched_fight") or {}, o.get("matched_quiet") or {}
        c = o.get("contrast") or {}
        rows = [
            f"## {title}",
            "",
            f"- fights={o.get('n_fight')} matched={o.get('n_matched')} coverage={fmt(o.get('coverage'), 3)}",
            f"- loaded matches={o.get('n_loaded')} cache_miss={o.get('n_cache_miss')} builder_fail={o.get('n_builder_fail')}",
            f"- unmatched reasons: `{json.dumps(o.get('unmatched_reason') or {})}`",
            "",
            "| Arm | n | matches | E[ΔV] | E[\\|ΔV\\|] | median \\|ΔV\\| | P(ΔV>0) |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Fight (matched) | {mf.get('n')} | {mf.get('n_matches')} | {fmt(mf.get('mean'))} | {fmt(mf.get('mean_abs'))} | {fmt(mf.get('median_abs'))} | {fmt(mf.get('p_pos'))} |",
            f"| Quiet (matched) | {mq.get('n')} | {mq.get('n_matches')} | {fmt(mq.get('mean'))} | {fmt(mq.get('mean_abs'))} | {fmt(mq.get('median_abs'))} | {fmt(mq.get('p_pos'))} |",
            "",
            "### Paired contrast (match-bootstrap)",
            "",
        ]
        for k, label in (
            ("mean_abs_fight_minus_quiet", "E[|ΔV|_fight − |ΔV|_quiet]"),
            ("mean_signed_fight_minus_quiet", "E[ΔV_fight − ΔV_quiet]"),
            ("p_pos_fight_minus_quiet", "P(ΔV>0)_fight − P(ΔV>0)_quiet"),
        ):
            d = (c or {}).get(k) or {}
            rows.append(
                f"- **{label}:** {fmt(d.get('estimate'), 5)}  CI95=[{fmt((d.get('ci95') or [None, None])[0], 5)}, "
                f"{fmt((d.get('ci95') or [None, None])[1], 5)}]  bootstrap_fraction_positive={fmt(d.get('bootstrap_fraction_positive'), 4)}"
            )
        bal = o.get("balance") or {}
        rows += [
            "",
            f"Balance: mean|Δp|={fmt(bal.get('mean_abs_dp'))} mean|Δt|={fmt(bal.get('mean_abs_dt_s'))}s "
            f"mean|ΔL|={fmt(bal.get('mean_abs_dL_ms'))}ms",
            "",
        ]
        return rows

    lines += block("Q_CAL (DEV matching / μ_Q, s_Q)", ca_out)
    lines += block("TEST 15.16 (primary quiet contrast)", te_out)
    lines += [
        "## Guardrails",
        "",
        "- Do not call quiet a causal control or independent fight-winner accuracy.",
        "- Do not feed L_i or quiet labels into q.",
        "- MAIN result uses new fit85 only; old_V quiet ratios forbidden.",
        "",
        f"Artifacts: `{OUT.as_posix()}/`",
        "",
    ]
    md = REPO / "docs" / "REVIEW_RESPONSE_RR3_QUIET_20260920.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # public slim JSON
    public = {
        k: payload[k]
        for k in ("generated", "epistemic", "design", "V_sha16", "matching", "interpretation")
    }
    public["Q_CAL"] = {k: ca_out.get(k) for k in (
        "n_fight", "n_matched", "coverage", "n_loaded", "n_cache_miss",
        "matched_fight", "matched_quiet", "contrast", "balance", "unmatched_reason",
    )}
    if te_out:
        public["TEST"] = {k: te_out.get(k) for k in (
            "n_fight", "n_matched", "coverage", "n_loaded", "n_cache_miss",
            "matched_fight", "matched_quiet", "contrast", "balance", "unmatched_reason",
        )}
    (REPO / "docs" / "REVIEW_RESPONSE_RR3_QUIET_20260920.json").write_text(
        json.dumps(scrub(public), indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", OUT / "quiet_contrast.json", md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
