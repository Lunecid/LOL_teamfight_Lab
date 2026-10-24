"""First-objective queries after component L (vectorized production + scalar reference)."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from . import config as C
from .parse import category_in_mask


def real_time_or_none(value: int) -> Optional[int]:
    """Convert saved sentinel times to None; do not treat -1/-2 as real timestamps."""
    v = int(value)
    if v == C.SENTINEL_MISSING or v == C.SENTINEL_INF or v < 0:
        return None
    return v


def restricted_upper_bound(
    next_kill_after_L: int,
    next_component_tau: int,
    game_end_ms: int,
    game_end_known: bool,
) -> tuple[Optional[int], str]:
    """Upper exclusive bound for pre-next-fight interval."""
    if not game_end_known:
        return None, "unknown_game_end"
    ge = real_time_or_none(game_end_ms)
    if ge is None:
        return None, "unknown_game_end"
    nk = real_time_or_none(next_kill_after_L)
    nt = real_time_or_none(next_component_tau)
    candidates = [ge]
    if nk is not None:
        candidates.append(nk)
    if nt is not None:
        candidates.append(nt)
    return int(min(candidates)), "ok"


def _tie_bitfield(teams: set[int]) -> int:
    tb = 0
    for tm in teams:
        if tm == C.TEAM_UNKNOWN:
            tb |= 1
        elif tm == C.TEAM_BLUE:
            tb |= 2
        elif tm == C.TEAM_RED:
            tb |= 4
        else:
            tb |= 8
    return int(tb)


def is_uncertain_acq(acq: int) -> bool:
    """True unknown/contradictory acquisition evidence (NOT explicit neutral 300)."""
    return int(acq) in (C.ACQ_UNKNOWN, C.ACQ_CONTRADICTORY)


def is_neutral_acq(acq: int) -> bool:
    return int(acq) == C.ACQ_NEUTRAL


def _precompute_category_index(
    obj_t: np.ndarray,
    obj_ord: np.ndarray,
    obj_mask: np.ndarray,
    obj_team: np.ndarray,
    obj_after_lf: np.ndarray,
    obj_acq: np.ndarray,
    category: str,
    mode: str,
) -> dict[str, np.ndarray]:
    """Filter/sort once per category; collapse same-ms ties to one representative row.

    mode:
      - 'confirmed': ACQ_CONFIRMED only (acquisition index)
      - 'uncertain': ACQ_UNKNOWN / ACQ_CONTRADICTORY (NOT neutral)
      - 'neutral': ACQ_NEUTRAL only (diagnostic; never acquisition or uncertain)
    """
    cat = str(category)
    bit = 1 << int(C.CATEGORY_TO_CODE[cat])
    keep = (np.asarray(obj_mask, dtype=np.int64) & bit) != 0
    acq = np.asarray(obj_acq, dtype=np.int8)
    if mode == "confirmed":
        keep &= acq == C.ACQ_CONFIRMED
    elif mode == "uncertain":
        keep &= (acq == C.ACQ_UNKNOWN) | (acq == C.ACQ_CONTRADICTORY)
    elif mode == "neutral":
        keep &= acq == C.ACQ_NEUTRAL
    else:
        raise ValueError(f"unknown category index mode: {mode}")
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        empty_i64 = np.zeros(0, dtype=np.int64)
        empty_i8 = np.zeros(0, dtype=np.int8)
        return {
            "t": empty_i64,
            "ord": empty_i64,
            "team": empty_i64,
            "after": empty_i8,
            "n_tie": empty_i64,
            "tie_mask": empty_i64,
            "tie_teams": empty_i64,
        }

    # Events already globally sorted by (t, ord); preserve that order.
    t = obj_t[idx].astype(np.int64, copy=False)
    ord_ = obj_ord[idx].astype(np.int64, copy=False)
    team = obj_team[idx].astype(np.int64, copy=False)
    after = obj_after_lf[idx].astype(np.int8, copy=False)
    mask = obj_mask[idx].astype(np.int64, copy=False)

    # Collapse identical timestamps: one representative (lowest event_ord).
    uniq_t: list[int] = []
    rep_ord: list[int] = []
    rep_team: list[int] = []
    rep_after: list[int] = []
    n_tie: list[int] = []
    tie_mask: list[int] = []
    tie_teams: list[int] = []

    i = 0
    n = int(t.size)
    while i < n:
        j = i + 1
        while j < n and int(t[j]) == int(t[i]):
            j += 1
        block = slice(i, j)
        ords = ord_[block]
        teams = team[block]
        afters = after[block]
        masks = mask[block]
        best_local = int(np.argmin(ords))
        uniq_t.append(int(t[i]))
        rep_ord.append(int(ords[best_local]))
        rep_team.append(int(teams[best_local]))
        rep_after.append(int(afters[best_local]))
        n_tie.append(int(j - i))
        tm = 0
        for m in masks:
            tm |= int(m)
        tie_mask.append(int(tm))
        tie_teams.append(_tie_bitfield({int(x) for x in teams.tolist()}))
        i = j

    return {
        "t": np.asarray(uniq_t, dtype=np.int64),
        "ord": np.asarray(rep_ord, dtype=np.int64),
        "team": np.asarray(rep_team, dtype=np.int64),
        "after": np.asarray(rep_after, dtype=np.int8),
        "n_tie": np.asarray(n_tie, dtype=np.int64),
        "tie_mask": np.asarray(tie_mask, dtype=np.int64),
        "tie_teams": np.asarray(tie_teams, dtype=np.int64),
    }


def first_objective_after_L(
    obj_t: np.ndarray,
    obj_ord: np.ndarray,
    obj_mask: np.ndarray,
    obj_team: np.ndarray,
    obj_after_lf: np.ndarray,
    obj_acq: np.ndarray,
    category: str,
    L: int,
    t_hi_inclusive: Optional[int],
    t_hi_exclusive: Optional[int],
) -> dict[str, Any]:
    """Scalar reference: first CONFIRMED acquisition in category with bound constraints.

    Explicit neutral (ACQ_NEUTRAL / killerTeamId=300) is confirmed absence of team
    credit — excluded from acquisition and from uncertain indexes. Neutral-only
    intervals are NO_ACQ. Unknown/contradictory evidence yields UNKNOWN_OBS when
    no confirmed event exists.
    """
    cat = str(category)
    L = int(L)
    out: dict[str, Any] = {
        "status": C.ST_NO_ACQ,
        "t_ms": C.SENTINEL_MISSING,
        "delay_ms": C.SENTINEL_MISSING,
        "event_ord": C.SENTINEL_MISSING,
        "team": C.TEAM_UNKNOWN,
        "same_ms_at_L": False,
        "after_last_frame": False,
        "n_tie_events": 0,
        "tie_category_mask": 0,
        "tie_teams": 0,
        "uncertain_precedes_confirmed": False,
        "uncertain_in_interval": False,
        "neutral_precedes_confirmed": False,
        "neutral_in_interval": False,
    }
    if obj_t.size == 0:
        return out

    def _in_bounds(t: int) -> bool:
        if t_hi_inclusive is not None and t > int(t_hi_inclusive):
            return False
        if t_hi_exclusive is not None and t >= int(t_hi_exclusive):
            return False
        return True

    # Diagnostic: confirmed category event exactly at L
    i_at = int(np.searchsorted(obj_t, L, side="left"))
    j_at = int(np.searchsorted(obj_t, L, side="right"))
    for k in range(i_at, j_at):
        if category_in_mask(int(obj_mask[k]), cat) and int(obj_acq[k]) == C.ACQ_CONFIRMED:
            out["same_ms_at_L"] = True
            break

    i0 = int(np.searchsorted(obj_t, L, side="right"))
    first_conf: Optional[int] = None
    first_unc: Optional[int] = None
    first_neut: Optional[int] = None
    for i in range(i0, obj_t.size):
        t = int(obj_t[i])
        if not _in_bounds(t):
            break
        if not category_in_mask(int(obj_mask[i]), cat):
            continue
        acq = int(obj_acq[i])
        if acq == C.ACQ_CONFIRMED:
            if first_conf is None:
                first_conf = i
                break
        elif is_uncertain_acq(acq):
            if first_unc is None:
                first_unc = i
        elif is_neutral_acq(acq):
            if first_neut is None:
                first_neut = i

    if first_conf is not None:
        t0 = int(obj_t[first_conf])
        tie_idx = []
        teams: set[int] = set()
        tie_mask = 0
        j = first_conf
        while j < obj_t.size and int(obj_t[j]) == t0:
            if category_in_mask(int(obj_mask[j]), cat):
                tie_mask |= int(obj_mask[j])
                if int(obj_acq[j]) == C.ACQ_CONFIRMED:
                    tie_idx.append(j)
                    teams.add(int(obj_team[j]))
            else:
                tie_mask |= int(obj_mask[j])
            j += 1
        j = int(np.searchsorted(obj_t, t0, side="left"))
        j2 = int(np.searchsorted(obj_t, t0, side="right"))
        for k in range(j, j2):
            tie_mask |= int(obj_mask[k])

        best = min(tie_idx, key=lambda k: int(obj_ord[k]))
        unc_before = False
        neut_before = False
        for i in range(i0, first_conf):
            t = int(obj_t[i])
            if t >= t0:
                break
            if not _in_bounds(t):
                break
            if not category_in_mask(int(obj_mask[i]), cat):
                continue
            acq = int(obj_acq[i])
            if is_uncertain_acq(acq):
                unc_before = True
            elif is_neutral_acq(acq):
                neut_before = True
        out["status"] = C.ST_OBSERVED
        out["t_ms"] = t0
        out["delay_ms"] = t0 - L
        out["event_ord"] = int(obj_ord[best])
        out["team"] = int(obj_team[best])
        out["after_last_frame"] = bool(obj_after_lf[best])
        out["n_tie_events"] = int(len(tie_idx))
        out["tie_category_mask"] = int(tie_mask)
        out["tie_teams"] = _tie_bitfield(teams)
        out["uncertain_precedes_confirmed"] = bool(unc_before)
        out["uncertain_in_interval"] = bool(unc_before or first_unc is not None)
        out["neutral_precedes_confirmed"] = bool(neut_before)
        out["neutral_in_interval"] = bool(neut_before or first_neut is not None)
        return out

    out["neutral_in_interval"] = first_neut is not None
    if first_unc is not None:
        out["status"] = C.ST_UNKNOWN_OBS
        out["uncertain_in_interval"] = True
        return out
    # Neutral-only (or empty) → confirmed absence of team credit → NO_ACQ
    return out


def query_component_intervals_scalar(
    obj_t: np.ndarray,
    obj_ord: np.ndarray,
    obj_mask: np.ndarray,
    obj_team: np.ndarray,
    obj_after_lf: np.ndarray,
    obj_acq: np.ndarray,
    L: int,
    game_end_ms: int,
    game_end_known: bool,
    next_kill_after_L: int,
    next_component_tau: int,
    categories: tuple[str, ...] = C.CATEGORY_ORDER,
) -> dict[str, dict[str, Any]]:
    """Scalar reference path (tests / equality checks)."""
    results: dict[str, dict[str, Any]] = {}
    L = int(L)

    if not game_end_known or real_time_or_none(game_end_ms) is None:
        for cat in categories:
            base = first_objective_after_L(
                obj_t, obj_ord, obj_mask, obj_team, obj_after_lf, obj_acq, cat, L, None, None
            )
            base["status"] = C.ST_UNKNOWN_OBS
            base["t_ms"] = C.SENTINEL_MISSING
            base["delay_ms"] = C.SENTINEL_MISSING
            results[f"{C.INTERVAL_UNCAPED}:{cat}"] = base
            no_int = dict(base)
            results[f"{C.INTERVAL_PRE_NEXT}:{cat}"] = no_int
        return results

    ge = int(game_end_ms)
    for cat in categories:
        results[f"{C.INTERVAL_UNCAPED}:{cat}"] = first_objective_after_L(
            obj_t, obj_ord, obj_mask, obj_team, obj_after_lf, obj_acq, cat, L, ge, None
        )

    nk = real_time_or_none(next_kill_after_L)
    nt = real_time_or_none(next_component_tau)
    if (nk is not None and nk <= L) or (nt is not None and nt <= L):
        for cat in categories:
            row = {
                "status": C.ST_NO_INTERVAL,
                "t_ms": C.SENTINEL_MISSING,
                "delay_ms": C.SENTINEL_MISSING,
                "event_ord": C.SENTINEL_MISSING,
                "team": C.TEAM_UNKNOWN,
                "same_ms_at_L": results[f"{C.INTERVAL_UNCAPED}:{cat}"]["same_ms_at_L"],
                "after_last_frame": False,
                "n_tie_events": 0,
                "tie_category_mask": 0,
                "tie_teams": 0,
                "uncertain_precedes_confirmed": False,
                "uncertain_in_interval": False,
                "neutral_precedes_confirmed": False,
                "neutral_in_interval": False,
            }
            results[f"{C.INTERVAL_PRE_NEXT}:{cat}"] = row
        return results

    ub, reason = restricted_upper_bound(next_kill_after_L, next_component_tau, ge, True)
    if ub is None or reason != "ok":
        for cat in categories:
            row = dict(results[f"{C.INTERVAL_UNCAPED}:{cat}"])
            row["status"] = C.ST_UNKNOWN_OBS
            row["t_ms"] = C.SENTINEL_MISSING
            row["delay_ms"] = C.SENTINEL_MISSING
            results[f"{C.INTERVAL_PRE_NEXT}:{cat}"] = row
        return results

    if int(ub) <= L:
        for cat in categories:
            row = {
                "status": C.ST_NO_INTERVAL,
                "t_ms": C.SENTINEL_MISSING,
                "delay_ms": C.SENTINEL_MISSING,
                "event_ord": C.SENTINEL_MISSING,
                "team": C.TEAM_UNKNOWN,
                "same_ms_at_L": results[f"{C.INTERVAL_UNCAPED}:{cat}"]["same_ms_at_L"],
                "after_last_frame": False,
                "n_tie_events": 0,
                "tie_category_mask": 0,
                "tie_teams": 0,
                "uncertain_precedes_confirmed": False,
                "uncertain_in_interval": False,
                "neutral_precedes_confirmed": False,
                "neutral_in_interval": False,
            }
            results[f"{C.INTERVAL_PRE_NEXT}:{cat}"] = row
        return results

    for cat in categories:
        results[f"{C.INTERVAL_PRE_NEXT}:{cat}"] = first_objective_after_L(
            obj_t, obj_ord, obj_mask, obj_team, obj_after_lf, obj_acq, cat, L, None, int(ub)
        )
    return results


# Back-compat alias used by older call sites / docs
query_component_intervals = query_component_intervals_scalar


def _apply_first_hit_vectorized(
    cat_idx: dict[str, np.ndarray],
    unc_idx: dict[str, np.ndarray],
    neut_idx: dict[str, np.ndarray],
    L_arr: np.ndarray,
    t_hi_inclusive: Optional[np.ndarray],
    t_hi_exclusive: Optional[np.ndarray],
    force_status: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Vectorized first-hit assignment for one category across all components in a match."""
    n = int(L_arr.size)
    status = np.full(n, C.ST_NO_ACQ, dtype=np.int8)
    t_ms = np.full(n, C.SENTINEL_MISSING, dtype=np.int64)
    delay = np.full(n, C.SENTINEL_MISSING, dtype=np.int64)
    eord = np.full(n, C.SENTINEL_MISSING, dtype=np.int64)
    team = np.full(n, C.TEAM_UNKNOWN, dtype=np.int16)
    same_at = np.zeros(n, dtype=np.int8)
    after = np.zeros(n, dtype=np.int8)
    n_tie = np.zeros(n, dtype=np.int16)
    tie_mask = np.zeros(n, dtype=np.int64)
    tie_teams = np.zeros(n, dtype=np.int8)
    unc_prec = np.zeros(n, dtype=np.int8)
    unc_in = np.zeros(n, dtype=np.int8)
    neut_prec = np.zeros(n, dtype=np.int8)
    neut_in = np.zeros(n, dtype=np.int8)

    ct = cat_idx["t"]
    if ct.size:
        i_left = np.searchsorted(ct, L_arr, side="left")
        i_right = np.searchsorted(ct, L_arr, side="right")
        same_at = (i_right > i_left).astype(np.int8)

        idx = np.searchsorted(ct, L_arr, side="right")
        valid = idx < ct.size
        t_cand = np.where(valid, ct[np.minimum(idx, ct.size - 1)], C.SENTINEL_MISSING)
        if t_hi_inclusive is not None:
            valid &= t_cand <= t_hi_inclusive
        if t_hi_exclusive is not None:
            valid &= t_cand < t_hi_exclusive

        safe_idx = np.minimum(idx, max(ct.size - 1, 0))
        t_ms = np.where(valid, ct[safe_idx], C.SENTINEL_MISSING)
        delay = np.where(valid, t_ms - L_arr, C.SENTINEL_MISSING)
        eord = np.where(valid, cat_idx["ord"][safe_idx], C.SENTINEL_MISSING)
        team = np.where(valid, cat_idx["team"][safe_idx], C.TEAM_UNKNOWN).astype(np.int16)
        after = np.where(valid, cat_idx["after"][safe_idx], 0).astype(np.int8)
        n_tie = np.where(valid, cat_idx["n_tie"][safe_idx], 0).astype(np.int16)
        tie_mask = np.where(valid, cat_idx["tie_mask"][safe_idx], 0)
        tie_teams = np.where(valid, cat_idx["tie_teams"][safe_idx], 0).astype(np.int8)
        status = np.where(valid, C.ST_OBSERVED, C.ST_NO_ACQ).astype(np.int8)

        ut = unc_idx["t"]
        if ut.size:
            uidx = np.searchsorted(ut, L_arr, side="right")
            uvalid = uidx < ut.size
            ut_cand = np.where(uvalid, ut[np.minimum(uidx, ut.size - 1)], C.SENTINEL_MISSING)
            if t_hi_inclusive is not None:
                uvalid &= ut_cand <= t_hi_inclusive
            if t_hi_exclusive is not None:
                uvalid &= ut_cand < t_hi_exclusive
            unc_in = uvalid.astype(np.int8)
            unc_prec = (uvalid & valid & (ut_cand < t_ms)).astype(np.int8)
            only_unc = uvalid & (~valid)
            status = np.where(only_unc, C.ST_UNKNOWN_OBS, status).astype(np.int8)

        nt = neut_idx["t"]
        if nt.size:
            nidx = np.searchsorted(nt, L_arr, side="right")
            nvalid = nidx < nt.size
            nt_cand = np.where(nvalid, nt[np.minimum(nidx, nt.size - 1)], C.SENTINEL_MISSING)
            if t_hi_inclusive is not None:
                nvalid &= nt_cand <= t_hi_inclusive
            if t_hi_exclusive is not None:
                nvalid &= nt_cand < t_hi_exclusive
            neut_in = nvalid.astype(np.int8)
            neut_prec = (nvalid & valid & (nt_cand < t_ms)).astype(np.int8)
    else:
        ut = unc_idx["t"]
        if ut.size:
            uidx = np.searchsorted(ut, L_arr, side="right")
            uvalid = uidx < ut.size
            ut_cand = np.where(uvalid, ut[np.minimum(uidx, ut.size - 1)], C.SENTINEL_MISSING)
            if t_hi_inclusive is not None:
                uvalid &= ut_cand <= t_hi_inclusive
            if t_hi_exclusive is not None:
                uvalid &= ut_cand < t_hi_exclusive
            unc_in = uvalid.astype(np.int8)
            status = np.where(uvalid, C.ST_UNKNOWN_OBS, C.ST_NO_ACQ).astype(np.int8)
        nt = neut_idx["t"]
        if nt.size:
            nidx = np.searchsorted(nt, L_arr, side="right")
            nvalid = nidx < nt.size
            nt_cand = np.where(nvalid, nt[np.minimum(nidx, nt.size - 1)], C.SENTINEL_MISSING)
            if t_hi_inclusive is not None:
                nvalid &= nt_cand <= t_hi_inclusive
            if t_hi_exclusive is not None:
                nvalid &= nt_cand < t_hi_exclusive
            neut_in = nvalid.astype(np.int8)

    if force_status is not None:
        forced = force_status
        status = forced.astype(np.int8)
        t_ms = np.full(n, C.SENTINEL_MISSING, dtype=np.int64)
        delay = np.full(n, C.SENTINEL_MISSING, dtype=np.int64)
        eord = np.full(n, C.SENTINEL_MISSING, dtype=np.int64)
        team = np.full(n, C.TEAM_UNKNOWN, dtype=np.int16)
        after = np.zeros(n, dtype=np.int8)
        n_tie = np.zeros(n, dtype=np.int16)
        tie_mask = np.zeros(n, dtype=np.int64)
        tie_teams = np.zeros(n, dtype=np.int8)
        unc_prec = np.zeros(n, dtype=np.int8)
        unc_in = np.zeros(n, dtype=np.int8)
        neut_prec = np.zeros(n, dtype=np.int8)
        neut_in = np.zeros(n, dtype=np.int8)

    return {
        "status": status,
        "t_ms": t_ms,
        "delay_ms": delay,
        "event_ord": eord,
        "team": team,
        "same_ms_at_L": same_at,
        "after_last_frame": after,
        "n_tie_events": n_tie,
        "tie_category_mask": tie_mask,
        "tie_teams": tie_teams,
        "uncertain_precedes_confirmed": unc_prec,
        "uncertain_in_interval": unc_in,
        "neutral_precedes_confirmed": neut_prec,
        "neutral_in_interval": neut_in,
    }


def query_match_components_vectorized(
    obj_t: np.ndarray,
    obj_ord: np.ndarray,
    obj_mask: np.ndarray,
    obj_team: np.ndarray,
    obj_after_lf: np.ndarray,
    obj_acq: np.ndarray,
    L_arr: np.ndarray,
    game_end_ms: int,
    game_end_known: bool,
    next_kill_after_L: np.ndarray,
    next_component_tau: np.ndarray,
    categories: Sequence[str] = C.CATEGORY_ORDER,
) -> dict[str, dict[str, np.ndarray]]:
    """Production path: one category filter/sort per match, searchsorted over all L."""
    L_arr = np.asarray(L_arr, dtype=np.int64)
    nk_arr = np.asarray(next_kill_after_L, dtype=np.int64)
    nt_arr = np.asarray(next_component_tau, dtype=np.int64)
    n = int(L_arr.size)
    results: dict[str, dict[str, np.ndarray]] = {}

    # Precompute per-category confirmed / uncertain / neutral indexes ONCE
    cat_indexes: dict[str, dict[str, np.ndarray]] = {}
    unc_indexes: dict[str, dict[str, np.ndarray]] = {}
    neut_indexes: dict[str, dict[str, np.ndarray]] = {}
    for cat in categories:
        cat_indexes[cat] = _precompute_category_index(
            obj_t, obj_ord, obj_mask, obj_team, obj_after_lf, obj_acq, cat, "confirmed"
        )
        unc_indexes[cat] = _precompute_category_index(
            obj_t, obj_ord, obj_mask, obj_team, obj_after_lf, obj_acq, cat, "uncertain"
        )
        neut_indexes[cat] = _precompute_category_index(
            obj_t, obj_ord, obj_mask, obj_team, obj_after_lf, obj_acq, cat, "neutral"
        )

    ge_known = bool(game_end_known) and real_time_or_none(game_end_ms) is not None
    if not ge_known:
        force = np.full(n, C.ST_UNKNOWN_OBS, dtype=np.int8)
        for cat in categories:
            hit = _apply_first_hit_vectorized(
                cat_indexes[cat],
                unc_indexes[cat],
                neut_indexes[cat],
                L_arr,
                None,
                None,
                force_status=force,
            )
            hit2 = _apply_first_hit_vectorized(
                cat_indexes[cat],
                unc_indexes[cat],
                neut_indexes[cat],
                L_arr,
                None,
                None,
                force_status=None,
            )
            hit["same_ms_at_L"] = hit2["same_ms_at_L"]
            results[f"{C.INTERVAL_UNCAPED}:{cat}"] = hit
            results[f"{C.INTERVAL_PRE_NEXT}:{cat}"] = {
                k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in hit.items()
            }
        return results

    ge = int(game_end_ms)
    ge_arr = np.full(n, ge, dtype=np.int64)
    for cat in categories:
        results[f"{C.INTERVAL_UNCAPED}:{cat}"] = _apply_first_hit_vectorized(
            cat_indexes[cat],
            unc_indexes[cat],
            neut_indexes[cat],
            L_arr,
            ge_arr,
            None,
            force_status=None,
        )

    # Restricted bounds per component
    nk_real = np.where(
        (nk_arr == C.SENTINEL_MISSING) | (nk_arr == C.SENTINEL_INF) | (nk_arr < 0),
        np.iinfo(np.int64).max,
        nk_arr,
    )
    nt_real = np.where(
        (nt_arr == C.SENTINEL_MISSING) | (nt_arr == C.SENTINEL_INF) | (nt_arr < 0),
        np.iinfo(np.int64).max,
        nt_arr,
    )
    ub = np.minimum(np.minimum(nk_real, nt_real), ge_arr)

    no_interval = (
        ((nk_arr >= 0) & (nk_arr <= L_arr))
        | ((nt_arr >= 0) & (nt_arr <= L_arr))
        | (ub <= L_arr)
    )

    for cat in categories:
        hit = _apply_first_hit_vectorized(
            cat_indexes[cat],
            unc_indexes[cat],
            neut_indexes[cat],
            L_arr,
            None,
            ub,
            force_status=None,
        )
        for key in ("t_ms", "delay_ms", "event_ord"):
            hit[key] = np.where(no_interval, C.SENTINEL_MISSING, hit[key])
        hit["team"] = np.where(no_interval, C.TEAM_UNKNOWN, hit["team"]).astype(np.int16)
        hit["after_last_frame"] = np.where(no_interval, 0, hit["after_last_frame"]).astype(np.int8)
        hit["n_tie_events"] = np.where(no_interval, 0, hit["n_tie_events"]).astype(np.int16)
        hit["tie_category_mask"] = np.where(no_interval, 0, hit["tie_category_mask"])
        hit["tie_teams"] = np.where(no_interval, 0, hit["tie_teams"]).astype(np.int8)
        hit["uncertain_precedes_confirmed"] = np.where(
            no_interval, 0, hit["uncertain_precedes_confirmed"]
        ).astype(np.int8)
        hit["uncertain_in_interval"] = np.where(
            no_interval, 0, hit["uncertain_in_interval"]
        ).astype(np.int8)
        hit["neutral_precedes_confirmed"] = np.where(
            no_interval, 0, hit["neutral_precedes_confirmed"]
        ).astype(np.int8)
        hit["neutral_in_interval"] = np.where(
            no_interval, 0, hit["neutral_in_interval"]
        ).astype(np.int8)
        hit["status"] = np.where(no_interval, C.ST_NO_INTERVAL, hit["status"]).astype(np.int8)
        hit["same_ms_at_L"] = results[f"{C.INTERVAL_UNCAPED}:{cat}"]["same_ms_at_L"]
        results[f"{C.INTERVAL_PRE_NEXT}:{cat}"] = hit
    return results


def classify_uncapped_vs_next_anchor(
    t_obj: int,
    status: int,
    L: int,
    next_kill_after_L: int,
    next_component_tau: int,
    game_end_ms: int,
    game_end_known: bool,
) -> str:
    """Classify where an uncapped first objective sits relative to next kill/anchor/terminal.

    A real next_component_tau <= L is an already-active/overlapping next component,
    consistent with restricted NO_INTERVAL — not 'no upcoming event'.
    """
    if status != C.ST_OBSERVED:
        return "no_observed_objective"
    t = int(t_obj)
    L = int(L)
    nk = real_time_or_none(next_kill_after_L)
    nt = real_time_or_none(next_component_tau)
    ge = real_time_or_none(game_end_ms) if game_end_known else None

    overlapping = (nt is not None and nt <= L) or (nk is not None and nk <= L)
    if overlapping:
        # Prefer the component-tau semantics highlighted by the design review.
        if nt is not None and nt <= L:
            return "already_active_overlapping_next_component"
        return "already_active_overlapping_next_kill"

    next_anchor = None
    for cand in (nk, nt):
        if cand is not None and cand > L:
            next_anchor = cand if next_anchor is None else min(next_anchor, cand)
    if next_anchor is not None:
        if t > next_anchor:
            return "after_next_kill_or_anchor"
        if t == next_anchor:
            return "same_ms_as_next_kill_or_anchor"
        if ge is not None and t == ge:
            return "at_terminal_before_next_anchor"
        return "before_next_kill_or_anchor"
    if ge is not None and t == ge:
        return "at_terminal_no_next_anchor"
    if ge is not None and t < ge:
        return "before_terminal_no_next_anchor"
    return "other"
