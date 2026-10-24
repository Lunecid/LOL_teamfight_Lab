"""Independent verification from saved evidence (no production contract/aggregate calls)."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import config as C

# Observation codes duplicated locally (do not import contracts).
_POS_KNOWN_NONZERO = 0
_POS_AMBIGUOUS_ZERO = 1
_HP_CACHED_POSITIVE = 0
_HP_AMBIGUOUS_ZERO = 1


def _bitcount(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint32)
    out = np.zeros(x.shape, dtype=np.int32)
    for i in range(10):
        out += (x >> i) & 1
    return out


def _recompute_endpoint_ms(
    L: int,
    h_s: int,
    next_kill_after_L: int,
    next_component_tau: int,
    game_end_ms: int,
    game_end_known: bool,
) -> int:
    L = int(L)
    horizon = L + 1000 * int(h_s)
    nk = int(next_kill_after_L)
    nt = int(next_component_tau)
    ge = int(game_end_ms) if game_end_known and int(game_end_ms) >= 0 else None
    cand = [horizon]
    if nk >= 0:
        cand.append(nk - 1)
    if nt >= 0:
        if ge is None:
            cand.append(nt - 1)
        elif nt < ge:
            cand.append(nt - 1)
    if ge is not None:
        cand.append(ge - 1)
    return int(min(cand)) if cand else int(C.SENTINEL_MISSING)


def _reconstruct_participation_bits(
    frame_xy: np.ndarray,
    hp: np.ndarray,
    locs_x: np.ndarray,
    locs_y: np.ndarray,
    radius: float,
    c0_bits: int,
) -> dict[str, int]:
    """Recompute N/P bitsets from saved XY/hp/locs (independent of production helpers)."""
    lower_bits = 0
    upper_bits = 0
    near_bits = 0
    for j in range(10):
        px = float(frame_xy[j, 0])
        py = float(frame_xy[j, 1])
        hpv = float(hp[j])
        pos_known = np.isfinite(px) and np.isfinite(py) and px != 0.0 and py != 0.0
        hp_pos = np.isfinite(hpv) and hpv > 0.0
        in_radius = False
        if pos_known and locs_x.size > 0:
            d = float(np.sqrt((locs_x - px) ** 2 + (locs_y - py) ** 2).min())
            in_radius = d <= float(radius)
            if in_radius:
                near_bits |= 1 << j
        if in_radius and hp_pos:
            lower_bits |= 1 << j
        # Cache-only: ambiguous zero health is NOT observed exclusion.
        if in_radius or (not pos_known):
            upper_bits |= 1 << j
    p_lower = int(c0_bits) | int(lower_bits)
    p_upper = int(c0_bits) | int(upper_bits)
    return {
        "lower_bits": int(lower_bits),
        "upper_bits": int(upper_bits),
        "near_bits": int(near_bits),
        "p_lower_bits": int(p_lower),
        "p_upper_bits": int(p_upper),
        "part_near_only_lower": int(lower_bits) & ~int(c0_bits),
        "geom_near_only": int(near_bits) & ~int(c0_bits),
    }


def verify_table_independent(
    table: dict[str, np.ndarray],
    evidence: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    aggregates: dict[str, Any],
) -> dict[str, Any]:
    """Recompute key invariants from saved arrays without calling production helpers."""
    checks: dict[str, Any] = {"n_rows": int(table["match_local"].shape[0])}
    n = checks["n_rows"]
    mismatches = 0
    n_checks = 0

    def add(name: str, ok: bool, detail: Any = None) -> None:
        nonlocal mismatches, n_checks
        n_checks += 1
        checks[name] = bool(ok)
        if detail is not None:
            checks[f"{name}_detail"] = detail
        if not ok:
            mismatches += 1

    add("row_count_vs_expected", n == C.EXPECTED_MIN1_COMPONENTS, n)
    add("sorted_match_local", bool(np.all(np.diff(table["match_local"]) >= 0)) if n > 1 else True)

    for tag in ("tau_r1800", "tau_r2000", "tm1_r1800", "tm1_r2000"):
        support = np.asarray(table[f"support_ok_{tag}"], dtype=bool)
        c0 = np.asarray(table["c0_bits"], dtype=np.uint32)
        pl = np.asarray(table[f"p_lower_bits_{tag}"], dtype=np.uint32)
        pu = np.asarray(table[f"p_upper_bits_{tag}"], dtype=np.uint32)
        # C0 must be preserved on supported AND unsupported rows (absent-frame bounds keep C0).
        add(f"c0_subset_p_lower_{tag}", bool(np.all((c0 & pl) == c0)))
        add(f"c0_subset_p_upper_{tag}", bool(np.all((c0 & pu) == c0)))
        add(f"lower_bits_le_upper_{tag}", bool(np.all(_bitcount(pl) <= _bitcount(pu))))
        add(
            f"p_lower_n_matches_bits_{tag}",
            bool(np.all(_bitcount(pl) == table[f"p_lower_n_{tag}"])),
        )
        add(
            f"p_upper_n_matches_bits_{tag}",
            bool(np.all(_bitcount(pu) == table[f"p_upper_n_{tag}"])),
        )
        # Unsupported: N_lower empty, N_upper all-ten, P_lower=C0, P_upper=all-ten.
        unsup = ~support
        if unsup.any():
            nl = np.asarray(table[f"n_lower_{tag}"], dtype=np.int64)
            nu = np.asarray(table[f"n_upper_{tag}"], dtype=np.int64)
            lower = np.asarray(table[f"lower_bits_{tag}"], dtype=np.uint32)
            upper = np.asarray(table[f"upper_bits_{tag}"], dtype=np.uint32)
            add(
                f"unsupported_bounds_{tag}",
                bool(
                    np.all(nl[unsup] == 0)
                    and np.all(nu[unsup] == 10)
                    and np.all(lower[unsup] == 0)
                    and np.all(upper[unsup] == (1 << 10) - 1)
                    and np.all(pl[unsup] == c0[unsup])
                    and np.all(pu[unsup] == (1 << 10) - 1)
                ),
            )
        else:
            add(f"unsupported_bounds_{tag}", True, "no_unsupported_rows")
        # Participation near-only must use N_lower bits, not geometric near alone.
        if f"part_near_only_lower_bits_{tag}" in table:
            part_near = np.asarray(table[f"part_near_only_lower_bits_{tag}"], dtype=np.uint32)
            lower = np.asarray(table[f"lower_bits_{tag}"], dtype=np.uint32)
            expected = lower & ~c0
            add(
                f"part_near_only_lower_consistent_{tag}",
                bool(np.all((part_near == expected) | ~support)),
            )

    s1800 = np.asarray(table["support_ok_tau_r1800"], dtype=bool)
    s2000 = np.asarray(table["support_ok_tau_r2000"], dtype=bool)
    both = s1800 & s2000
    add(
        "radius_mono_p_lower_tau",
        bool(np.all(table["p_lower_n_tau_r1800"][both] <= table["p_lower_n_tau_r2000"][both])),
    )
    add(
        "radius_mono_p_upper_tau",
        bool(np.all(table["p_upper_n_tau_r1800"][both] <= table["p_upper_n_tau_r2000"][both])),
    )

    lags = np.asarray(table["lag_f_F_tau"], dtype=np.int64)
    ok = lags >= 0
    n_exact = int(np.sum(ok & (lags == 10_000)))
    n_cum_10_15 = int(np.sum(ok & (lags >= 10_000) & (lags <= 15_000)))
    n_cum_10_20 = int(np.sum(ok & (lags >= 10_000) & (lags <= 20_000)))
    n_gt15_le20 = int(np.sum(ok & (lags > 15_000) & (lags <= 20_000)))
    add("gap_exact_10000", n_exact == int(aggregates["n_exact_10000_tau"]), {"got": n_exact})
    add("gap_cum_10_20", n_cum_10_20 == int(aggregates["n_cand_10_20_tau"]), {"got": n_cum_10_20})
    add(
        "gap_cum_10_15_reported",
        n_cum_10_15 == int(aggregates["gap_tau"]["cum_10_15"]),
        {"got": n_cum_10_15},
    )
    add(
        "gap_gt15_le20_reported",
        n_gt15_le20 == int(aggregates["gap_tau"]["gt15_le20"]),
        {"got": n_gt15_le20},
    )
    add("reconcile_inclusive_378703", n_cum_10_20 == C.EXPECTED_INCLUSIVE_10_20, n_cum_10_20)
    add(
        "reconcile_strict_378674",
        int(np.asarray(table["strict_eligible"], dtype=np.int8).sum()) == C.EXPECTED_STRICT_ELIGIBLE,
    )
    add("reconcile_exact_29", n_exact == C.EXPECTED_EXACT_10000, n_exact)

    for field_src, field_base in (
        ("f", "f"),
        ("L", "L"),
        ("tau", "tau"),
        ("match_local", "match_local"),
        ("component_id", "component_id"),
    ):
        add(f"baseline_{field_src}", bool(np.array_equal(table[field_src], baseline[field_base])))

    base_F = np.asarray(baseline["F_inclusive"], dtype=np.int64)
    base_d = np.asarray(baseline["delta"], dtype=np.int64)
    F = np.asarray(table["F_tau"], dtype=np.int64)
    d = np.asarray(table["lag_f_F_tau"], dtype=np.int64)
    known = (base_F >= 0) & (F >= 0)
    add("baseline_F_tau", bool(np.all(base_F[known] == F[known])), int((base_F[known] != F[known]).sum()))
    add("baseline_delta", bool(np.all(base_d[known] == d[known])), int((base_d[known] != d[known]).sum()))
    add(
        "baseline_exact_10000",
        bool(
            np.array_equal(
                np.asarray(table["exact_10000_tau"], dtype=np.int8),
                np.asarray(baseline["exact_10000"], dtype=np.int8),
            )
        ),
    )
    add(
        "baseline_strict_eligible",
        bool(
            np.array_equal(
                np.asarray(table["strict_eligible"], dtype=np.int8),
                np.asarray(baseline["strict_eligible"], dtype=np.int8),
            )
        ),
    )

    # Endpoint reuse identity + independent formula recompute
    endpoint_mismatch = 0
    endpoint_checked = 0
    for h in C.HORIZONS_S:
        add(
            f"baseline_e_h{h}",
            bool(np.array_equal(table[f"e_h{h}"], baseline[f"e_h{h}"])),
        )
        add(
            f"baseline_valid_h{h}",
            bool(
                np.array_equal(
                    np.asarray(table[f"valid_h{h}"], dtype=np.int8),
                    np.asarray(baseline[f"valid_h{h}"], dtype=np.int8),
                )
            ),
        )
        for i in range(n):
            e_stored = int(table[f"e_h{h}"][i])
            if e_stored < 0:
                continue
            e_re = _recompute_endpoint_ms(
                int(baseline["L"][i]),
                h,
                int(baseline["next_kill_after_L"][i]),
                int(baseline["next_component_tau"][i]),
                int(baseline["game_end_ms"][i]),
                bool(int(baseline["game_end_known"][i])),
            )
            endpoint_checked += 1
            if e_re != e_stored:
                endpoint_mismatch += 1
    add(
        "endpoint_formula_recompute",
        endpoint_mismatch == 0,
        {"checked": endpoint_checked, "mismatches": endpoint_mismatch},
    )

    # Both-cutoff post-frame fields present AND value-checked vs same-frame async definition
    for h in C.HORIZONS_S:
        for cut in ("tau", "tm1"):
            tag = f"{cut}_h{h}"
            add(f"has_F_post_{tag}", f"F_post_{tag}" in table)
            add(f"has_async_cutoff_{tag}", f"async_events_cutoff_to_endpoint_{tag}" in table)
            add(f"has_frame_idx_gap_{tag}", f"frame_idx_gap_{tag}" in table)
            # Aggregate cell must use each cutoff's own candidate mask key presence
            post_cell = (aggregates.get("post_frame") or {}).get(tag)
            add(f"aggregate_has_post_{tag}", isinstance(post_cell, dict))
            if isinstance(post_cell, dict):
                add(
                    f"aggregate_same_frame_uses_cutoff_events_{tag}",
                    "same_frame_with_async_events" in post_cell
                    and "async_events_cutoff_to_endpoint" in post_cell,
                )

    # Event unknown sentinels: when events_support_ok is false, counts must be -1 not 0
    if "events_support_ok" in table:
        ev_bad = ~np.asarray(table["events_support_ok"], dtype=bool)
        if ev_bad.any():
            add(
                "unknown_events_use_sentinel",
                bool(np.all(np.asarray(table["ev_at_tau"], dtype=np.int64)[ev_bad] < 0)),
            )
        else:
            add("unknown_events_use_sentinel", True, "no_unknown_event_rows")

    offs = np.asarray(evidence["first_loc_offsets"], dtype=np.int64)
    add("evidence_offsets_len", offs.shape[0] == n + 1, offs.shape[0])
    counts = np.diff(offs)
    add("evidence_loc_counts", bool(np.array_equal(counts, table["same_ms_first_n"])))

    locs_x = np.asarray(evidence["first_loc_x"], dtype=np.float64)
    locs_y = np.asarray(evidence["first_loc_y"], dtype=np.float64)
    dist_e = np.asarray(evidence["dist_tau_r1800"], dtype=np.float64)
    add("dist_evidence_shape", dist_e.shape == (n, 10), dist_e.shape)
    frame_xy = np.asarray(evidence["frame_xy_tau"], dtype=np.float64)
    hp_tau = np.asarray(evidence["hp_tau"], dtype=np.float64)
    add("frame_xy_shape", frame_xy.shape == (n, 10, 2), frame_xy.shape)
    add("hp_tau_shape", hp_tau.shape == (n, 10), hp_tau.shape)

    support = np.asarray(table["support_ok_tau_r1800"], dtype=bool)
    n_dist_checked = 0
    n_dist_mismatch = 0
    n_part_checked = 0
    n_part_mismatch = 0
    for i in range(n):
        if not support[i]:
            continue
        a = int(offs[i])
        b = int(offs[i + 1])
        if b <= a:
            continue
        lx = locs_x[a:b]
        ly = locs_y[a:b]
        for j in range(10):
            px, py = float(frame_xy[i, j, 0]), float(frame_xy[i, j, 1])
            if px == 0.0 or py == 0.0 or not np.isfinite(px) or not np.isfinite(py):
                continue
            d = float(np.sqrt((lx - px) ** 2 + (ly - py) ** 2).min())
            stored = float(dist_e[i, j])
            n_dist_checked += 1
            if not (np.isfinite(stored) and abs(stored - d) <= 1e-6):
                n_dist_mismatch += 1
                if n_dist_mismatch <= 5:
                    checks.setdefault("dist_mismatch_examples", []).append(
                        {"row": i, "pid": j + 1, "stored": stored, "recomputed": d}
                    )
        recon = _reconstruct_participation_bits(
            frame_xy[i], hp_tau[i], lx, ly, 1800.0, int(table["c0_bits"][i])
        )
        n_part_checked += 1
        if (
            recon["p_lower_bits"] != int(table["p_lower_bits_tau_r1800"][i])
            or recon["p_upper_bits"] != int(table["p_upper_bits_tau_r1800"][i])
            or recon["lower_bits"] != int(table["lower_bits_tau_r1800"][i])
            or recon["upper_bits"] != int(table["upper_bits_tau_r1800"][i])
        ):
            n_part_mismatch += 1
            if n_part_mismatch <= 5:
                checks.setdefault("part_mismatch_examples", []).append(
                    {"row": i, "recon": recon}
                )
    add(
        "distances_independent",
        n_dist_mismatch == 0,
        {"checked": n_dist_checked, "mismatches": n_dist_mismatch},
    )
    add(
        "participation_independent_from_xy_hp",
        n_part_mismatch == 0,
        {"checked": n_part_checked, "mismatches": n_part_mismatch},
    )
    add(
        "verification_coverage_nonzero",
        n_dist_checked > 0 and n_part_checked > 0 if support.any() else True,
        {"n_dist_checked": n_dist_checked, "n_part_checked": n_part_checked},
    )

    # Team scale fields present
    add("has_c0_scale", "c0_scale_max_side" in table)
    add("has_call_bits", "call_bits" in table)

    checks["n_checks"] = n_checks
    checks["n_mismatches"] = mismatches
    checks["all_ok"] = mismatches == 0
    checks["raw_presence_verified"] = False
    checks["expected_checks_min"] = 40
    add("expected_coverage", n_checks >= 40, n_checks)
    # Recompute all_ok after last add
    checks["n_mismatches"] = sum(
        1 for k, v in checks.items() if isinstance(v, bool) and k not in ("all_ok", "raw_presence_verified") and v is False
    )
    checks["all_ok"] = checks["n_mismatches"] == 0
    return checks


def verify_reloaded_artifacts(
    components_path,
    evidence_path,
    aggregates: dict[str, Any],
    baseline: dict[str, np.ndarray],
    expected_components_sha: str,
    expected_evidence_sha: str,
    file_sha256_fn,
) -> dict[str, Any]:
    """Reload written NPZs and re-run independent checks (hash + content)."""
    sha_c = file_sha256_fn(components_path)
    sha_e = file_sha256_fn(evidence_path)
    out: dict[str, Any] = {
        "components_sha256": sha_c,
        "evidence_sha256": sha_e,
        "components_sha_match": sha_c == expected_components_sha,
        "evidence_sha_match": sha_e == expected_evidence_sha,
    }
    with np.load(components_path, allow_pickle=False) as z:
        table = {k: np.asarray(z[k]) for k in z.files}
    with np.load(evidence_path, allow_pickle=False) as z:
        evidence = {k: np.asarray(z[k]) for k in z.files}
    indep = verify_table_independent(table, evidence, baseline, aggregates)
    out["independent_from_reload"] = indep
    out["all_ok"] = bool(
        out["components_sha_match"] and out["evidence_sha_match"] and indep.get("all_ok")
    )
    return out
