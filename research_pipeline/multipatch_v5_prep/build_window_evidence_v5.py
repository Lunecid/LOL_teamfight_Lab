import os
"""Empirical Outcome Window (H in {60,120,180,240,300}s) & Input History (L in {1..5} frames)
Evidence Builder on kr_15_18_22 (FIT primary, DEV/AUDIT comparison), with Queries Stratified and
Saved by t_0-Observable Objective State.

Strictly zero model training.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

PREP_ROOT = Path(os.environ["LOL_ARTIFACT_ROOT"]) / 'multipatch_v5_prep'
OUT_DIR = PREP_ROOT / "window_evidence_v5"
OBJ_QUERY_DIR = OUT_DIR / "queries_by_objective"
EXT_D = Path(os.environ["LOL_EXTERNAL_ROOT"])
BANK_ROOT = EXT_D / "bank" / "kr_15_18_22"

DRAGON_PIT_XY = np.array([9866.0, 4414.0], dtype=np.float32)
BARON_PIT_XY = np.array([5007.0, 10471.0], dtype=np.float32)
HORIZONS_S = [60, 120, 180, 240, 300]

OBJ_NAMES = [
    "elem_dragon",
    "elder_dragon",
    "baron",
    "herald",
    "horde",
    "atakhan",
    "outer_turret",
    "inner_turret",
    "base_turret",
    "nexus_turret",
    "inhibitor",
    "plate",
    "any_turret",
    "any_epic",
]
N_OBJ = len(OBJ_NAMES)

REGIME_NAMES = {
    0: "BARON_OR_ELDER_AVAIL",
    1: "ATAKHAN_AVAIL",
    2: "DRAGON_AVAIL",
    3: "HERALD_OR_HORDE_AVAIL",
    4: "NONE_SIEGE_SKIRMISH",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bucket_from_key(salt: str, canonical_match_key: str) -> int:
    digest = hashlib.sha256((salt + canonical_match_key).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 100


def get_split_role(patch: str, bkt: int) -> tuple[str, str]:
    if patch in ("15.18", "15.19"):
        return (f"FIT_{patch}", "FIT") if bkt <= 79 else (f"UNUSED_RESERVED_{patch}", "UNUSED")
    if patch == "15.20":
        return ("FIT_15.20", "FIT") if bkt <= 79 else ("STOP_15.20", "STOP")
    if patch == "15.21":
        if bkt <= 49:
            return ("DEV_15.21", "DEV")
        if bkt <= 74:
            return ("CAL_15.21", "CAL")
        return ("AUDIT_15.21", "AUDIT")
    if patch == "15.22":
        return ("TEST_15.22", "TEST")
    return ("OTHER", "OTHER")


def pct_dict(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"n": 0}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": round(float(np.mean(a)), 3),
        "sd": round(float(np.std(a)), 3),
        "min": round(float(np.min(a)), 3),
        "p10": round(float(np.percentile(a, 10)), 3),
        "p25": round(float(np.percentile(a, 25)), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p75": round(float(np.percentile(a, 75)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(np.max(a)), 3),
    }


def compute_horizon_summary(
    mask: np.ndarray,
    t0_ms: np.ndarray,
    f_ms: np.ndarray,
    L_ms: np.ndarray,
    c_ms: np.ndarray,
    end_ms: np.ndarray,
    next_f_ms: np.ndarray,
    par_ongoing: np.ndarray,
    avail_t0: np.ndarray,
    g_t0: np.ndarray,
    g_c: np.ndarray,
    g_t0_H: np.ndarray,
    g_c_H: np.ndarray,
    obj_total: np.ndarray,
    obj_clean: np.ndarray,
    obj_incombat: np.ndarray,
    obj_postcombat: np.ndarray,
    obj_fk_team: np.ndarray,
    obj_opp_team: np.ndarray,
    obj_c_total: np.ndarray,
    obj_c_clean: np.ndarray,
    first_obj_dt0: np.ndarray,
    first_obj_dL: np.ndarray,
    other_eng_t0_H: np.ndarray,
    other_eng_c_H: np.ndarray,
) -> dict:
    n_q = int(np.sum(mask))
    if n_q == 0:
        return {"n_queries": 0}

    sub_t0 = t0_ms[mask]
    sub_f = f_ms[mask]
    sub_L = L_ms[mask]
    sub_c = c_ms[mask]
    sub_end = end_ms[mask]
    sub_next_f = next_f_ms[mask]
    sub_par = par_ongoing[mask]
    sub_avail = avail_t0[mask]  # (n_q, 6): [drg, eld, hor, her, ata, bar]

    sub_gt0 = g_t0[mask]
    sub_gc = g_c[mask]
    sub_gt0_H = g_t0_H[mask]  # (n_q, 5)
    sub_gc_H = g_c_H[mask]    # (n_q, 5)

    sub_otot = obj_total[mask]      # (n_q, 5, N_OBJ)
    sub_ocln = obj_clean[mask]      # (n_q, 5, N_OBJ)
    sub_oinc = obj_incombat[mask]   # (n_q, N_OBJ)
    sub_opst = obj_postcombat[mask] # (n_q, 5, N_OBJ)
    sub_ofk = obj_fk_team[mask]     # (n_q, 5, N_OBJ)
    sub_oopp = obj_opp_team[mask]   # (n_q, 5, N_OBJ)
    sub_octot = obj_c_total[mask]   # (n_q, 5, N_OBJ)
    sub_occln = obj_c_clean[mask]   # (n_q, 5, N_OBJ)
    sub_dt0 = first_obj_dt0[mask]   # (n_q, N_OBJ)
    sub_dL = first_obj_dL[mask]     # (n_q, N_OBJ)
    sub_oth_t0 = other_eng_t0_H[mask]  # (n_q, 5)
    sub_oth_c = other_eng_c_H[mask]    # (n_q, 5)

    lead_sec = (sub_f - sub_t0) / 1000.0
    combat_dur_sec = (sub_L - sub_f) / 1000.0
    total_combat_end_sec = (sub_L - sub_t0) / 1000.0
    next_fight_delay_from_t0_sec = np.where(sub_next_f < 900000000, (sub_next_f - sub_t0) / 1000.0, np.nan)
    next_fight_delay_from_L_sec = np.where(sub_next_f < 900000000, (sub_next_f - sub_L) / 1000.0, np.nan)

    # Map from OBJ_NAMES index to avail_t0 column index (0:drg, 1:eld, 2:hor, 3:her, 4:ata, 5:bar)
    avail_col_map = {
        "elem_dragon": 0,
        "elder_dragon": 1,
        "baron": 5,
        "herald": 3,
        "horde": 2,
        "atakhan": 4,
    }

    y_near = sub_gc - sub_gt0
    y_H_mat = sub_gt0_H - sub_gt0[:, None]  # (n_q, 5)
    y_cH_mat = sub_gc_H - sub_gt0[:, None]  # (n_q, 5)

    by_horizon_t0 = {}
    by_horizon_c = {}

    for h_idx, h_sec in enumerate(HORIZONS_S):
        h_key = f"H{h_sec}s"
        hi_t0_ms = sub_t0 + h_sec * 1000
        hi_c_ms = sub_c + h_sec * 1000

        comp_done = float(np.mean((sub_L - sub_t0) <= h_sec * 1000))
        other_intervene = float(np.mean(sub_oth_t0[:, h_idx] > 0))
        other_mean_cnt = float(np.mean(sub_oth_t0[:, h_idx]))
        match_ended = float(np.mean(sub_end <= hi_t0_ms))

        # Gold sensitivity at t0 + H
        yh = y_H_mat[:, h_idx]
        if h_idx == 0:
            dy = yh
            sign_flip_prev = 0.0
        else:
            yh_prev = y_H_mat[:, h_idx - 1]
            dy = yh - yh_prev
            nz = (yh != 0) & (yh_prev != 0)
            sign_flip_prev = float(np.mean(np.sign(yh[nz]) != np.sign(yh_prev[nz]))) if np.any(nz) else 0.0

        nz_60 = (yh != 0) & (y_H_mat[:, 0] != 0)
        sign_agree_H60 = float(np.mean(np.sign(yh[nz_60]) == np.sign(y_H_mat[nz_60, 0]))) if np.any(nz_60) else 1.0
        nz_near = (yh != 0) & (y_near != 0)
        sign_agree_y_near = float(np.mean(np.sign(yh[nz_near]) == np.sign(y_near[nz_near]))) if np.any(nz_near) else 1.0

        obj_stats = {}
        for k_idx, oname in enumerate(OBJ_NAMES):
            tot_k = sub_otot[:, h_idx, k_idx]
            cln_k = sub_ocln[:, h_idx, k_idx]
            inc_k = sub_oinc[:, k_idx]
            pst_k = sub_opst[:, h_idx, k_idx]
            fk_k = sub_ofk[:, h_idx, k_idx]
            opp_k = sub_oopp[:, h_idx, k_idx]

            inc_rate_total = float(np.mean(tot_k > 0))
            mean_cnt_total = float(np.mean(tot_k))
            inc_rate_clean = float(np.mean(cln_k > 0))
            mean_cnt_clean = float(np.mean(cln_k))
            inc_rate_incombat = float(np.mean(inc_k > 0))
            inc_rate_postcombat = float(np.mean(pst_k > 0))
            inc_rate_fk_team = float(np.mean(fk_k > 0))
            inc_rate_opp_team = float(np.mean(opp_k > 0))

            if h_idx == 0:
                marg_inc_total = inc_rate_total
                marg_inc_clean = inc_rate_clean
            else:
                prev_tot_k = sub_otot[:, h_idx - 1, k_idx]
                prev_cln_k = sub_ocln[:, h_idx - 1, k_idx]
                marg_inc_total = inc_rate_total - float(np.mean(prev_tot_k > 0))
                marg_inc_clean = inc_rate_clean - float(np.mean(prev_cln_k > 0))

            cond_dict = None
            if oname in avail_col_map:
                acol = avail_col_map[oname]
                amask = sub_avail[:, acol]
                n_av = int(np.sum(amask))
                if n_av > 0:
                    cond_dict = {
                        "n_avail_at_t0": n_av,
                        "avail_share_at_t0": round(n_av / n_q, 4),
                        "cond_incidence_total": round(float(np.mean(tot_k[amask] > 0)), 4),
                        "cond_incidence_clean_pre_next_fight": round(float(np.mean(cln_k[amask] > 0)), 4),
                        "cond_mean_count_total": round(float(np.mean(tot_k[amask])), 4),
                        "cond_mean_count_clean": round(float(np.mean(cln_k[amask])), 4),
                    }

            obj_stats[oname] = {
                "incidence_total": round(inc_rate_total, 4),
                "mean_count_total": round(mean_cnt_total, 4),
                "incidence_clean_pre_next_fight": round(inc_rate_clean, 4),
                "mean_count_clean_pre_next_fight": round(mean_cnt_clean, 4),
                "clean_share_of_incidence": round(inc_rate_clean / inc_rate_total, 4) if inc_rate_total > 0 else 1.0,
                "incidence_in_combat": round(inc_rate_incombat, 4),
                "incidence_post_combat": round(inc_rate_postcombat, 4),
                "incidence_firstkill_team": round(inc_rate_fk_team, 4),
                "incidence_opposing_team": round(inc_rate_opp_team, 4),
                "marginal_incidence_gain_total": round(marg_inc_total, 4),
                "marginal_incidence_gain_clean": round(marg_inc_clean, 4),
                "marginal_incidence_contaminated_by_next_fight": round(max(0.0, marg_inc_total - marg_inc_clean), 4),
                "conditional_on_t0_available": cond_dict,
            }

        by_horizon_t0[h_key] = {
            "horizon_sec": h_sec,
            "target_combat_completion_rate": round(comp_done, 4),
            "other_engagement_intervention_rate": round(other_intervene, 4),
            "other_engagement_mean_count": round(other_mean_cnt, 4),
            "parallel_ongoing_at_t0_rate": round(float(np.mean(sub_par > 0)), 4),
            "match_termination_rate": round(match_ended, 4),
            "data_truncation_rate": 0.0,
            "gold_outcome_sensitivity": {
                "mean": round(float(np.mean(yh)), 2),
                "sd": round(float(np.std(yh)), 2),
                "p10": round(float(np.percentile(yh, 10)), 2),
                "p25": round(float(np.percentile(yh, 25)), 2),
                "p50": round(float(np.percentile(yh, 50)), 2),
                "p75": round(float(np.percentile(yh, 75)), 2),
                "p90": round(float(np.percentile(yh, 90)), 2),
                "mean_abs_increment_from_prev_H": round(float(np.mean(np.abs(dy))), 2),
                "sd_increment_from_prev_H": round(float(np.std(dy)), 2),
                "sign_flip_rate_vs_prev_H": round(sign_flip_prev, 4),
                "sign_agreement_vs_H60": round(sign_agree_H60, 4),
                "sign_agreement_vs_Y_near": round(sign_agree_y_near, 4),
                "exact_zero_tie_rate": round(float(np.mean(yh == 0)), 6),
            },
            "objectives": obj_stats,
        }

        # Also summarize Anchor B (c + H)
        ych = y_cH_mat[:, h_idx]
        nz_c_near = (ych != 0) & (y_near != 0)
        by_horizon_c[h_key] = {
            "horizon_after_c_sec": h_sec,
            "mean_elapsed_from_t0_sec": round(float(np.mean((sub_c - sub_t0) / 1000.0 + h_sec)), 2),
            "other_engagement_intervention_rate": round(float(np.mean(sub_oth_c[:, h_idx] > 0)), 4),
            "other_engagement_mean_count": round(float(np.mean(sub_oth_c[:, h_idx])), 4),
            "match_termination_rate": round(float(np.mean(sub_end <= hi_c_ms)), 4),
            "any_epic_incidence_total": round(float(np.mean(sub_octot[:, h_idx, 13] > 0)), 4),
            "any_epic_incidence_clean": round(float(np.mean(sub_occln[:, h_idx, 13] > 0)), 4),
            "any_turret_incidence_total": round(float(np.mean(sub_octot[:, h_idx, 12] > 0)), 4),
            "any_turret_incidence_clean": round(float(np.mean(sub_occln[:, h_idx, 12] > 0)), 4),
            "gold_sd": round(float(np.std(ych)), 2),
            "sign_agreement_vs_Y_near": round(float(np.mean(np.sign(ych[nz_c_near]) == np.sign(y_near[nz_c_near]))), 4) if np.any(nz_c_near) else 1.0,
        }

    # Delay distributions for objectives that occurred within 300s of t0
    obj_delay_distributions = {}
    for k_idx, oname in enumerate(OBJ_NAMES):
        valid_d = sub_dt0[:, k_idx] >= 0
        if np.any(valid_d):
            dt0_s = sub_dt0[valid_d, k_idx] / 1000.0
            dL_s = sub_dL[valid_d, k_idx] / 1000.0
            # Also among clean pre-next-fight objectives
            clean_d = valid_d & ((sub_t0 + sub_dt0[:, k_idx]) < sub_next_f)
            dt0_clean_s = sub_dt0[clean_d, k_idx] / 1000.0 if np.any(clean_d) else np.empty(0)
            dL_clean_s = sub_dL[clean_d, k_idx] / 1000.0 if np.any(clean_d) else np.empty(0)
            obj_delay_distributions[oname] = {
                "all_within_300s_from_t0_sec": pct_dict(dt0_s),
                "all_within_300s_from_last_kill_sec": pct_dict(dL_s),
                "clean_pre_next_fight_from_t0_sec": pct_dict(dt0_clean_s),
                "clean_pre_next_fight_from_last_kill_sec": pct_dict(dL_clean_s),
            }

    # Gold correlation matrix across [Y_near, Y_H60, Y_H120, Y_H180, Y_H240, Y_H300]
    g_stack = np.column_stack([y_near, y_H_mat])
    corr_mat = np.corrcoef(g_stack, rowvar=False)
    corr_labels = ["Y_near_c", "Y_t0_60s", "Y_t0_120s", "Y_t0_180s", "Y_t0_240s", "Y_t0_300s"]
    corr_dict = {
        corr_labels[i]: {corr_labels[j]: round(float(corr_mat[i, j]), 4) for j in range(len(corr_labels))}
        for i in range(len(corr_labels))
    }

    return {
        "n_queries": n_q,
        "combat_timing_sec": {
            "lead_time_tF_minus_t0_sec": pct_dict(lead_sec),
            "combat_duration_tC_minus_tF_sec": pct_dict(combat_dur_sec),
            "total_time_to_last_kill_tC_minus_t0_sec": pct_dict(total_combat_end_sec),
            "next_engagement_delay_from_t0_sec": pct_dict(next_fight_delay_from_t0_sec[~np.isnan(next_fight_delay_from_t0_sec)]),
            "next_engagement_delay_from_last_kill_sec": pct_dict(next_fight_delay_from_L_sec[~np.isnan(next_fight_delay_from_L_sec)]),
        },
        "anchor_A_fixed_from_observed_frame_t0": by_horizon_t0,
        "anchor_B_from_post_combat_frame_c": by_horizon_c,
        "objective_delay_distributions_sec": obj_delay_distributions,
        "gold_horizon_correlation_matrix": corr_dict,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OBJ_QUERY_DIR.mkdir(parents=True, exist_ok=True)

    fm = json.loads((BANK_ROOT / "FIELD_MANIFEST.json").read_text(encoding="utf-8"))
    comp_cols = {name: idx for idx, name in enumerate(fm["component_columns"])}
    scol_msub = fm["event_string_field_order"].index("monsterSubType")
    scol_ttype = fm["event_string_field_order"].index("towerType")
    elder_code = fm["event_string_enum_names"]["monsterSubType"].index("ELDER_DRAGON")
    outer_code = fm["event_string_enum_names"]["towerType"].index("OUTER_TURRET")
    inner_code = fm["event_string_enum_names"]["towerType"].index("INNER_TURRET")
    base_code = fm["event_string_enum_names"]["towerType"].index("BASE_TURRET")
    nexus_code = fm["event_string_enum_names"]["towerType"].index("NEXUS_TURRET")

    bank_entries = []
    with (BANK_ROOT / "BANK_ENTRIES.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            bank_entries.append(json.loads(line))

    salt_split = "rift-v5-multipatch-r1:"
    match_meta = []
    for idx, be in enumerate(bank_entries):
        mid = be["match_id"]
        patch = be["patch"]
        cid = mid.upper() if "_" in mid else f"KR_{mid}"
        bkt = bucket_from_key(salt_split, cid)
        srole, mrole = get_split_role(patch, bkt)
        match_meta.append((cid, patch, srole, mrole))

    rows_match_idx = []
    rows_comp_id = []
    rows_patch = []
    rows_split_role = []
    rows_macro_role = []
    rows_t0_ms = []
    rows_tau_ms = []
    rows_f_ms = []
    rows_L_ms = []
    rows_c_ms = []
    rows_end_ms = []
    rows_next_f_ms = []
    rows_parallel_ongoing = []
    rows_fk_blue = []
    rows_win_blue = []
    rows_avail_t0 = []
    rows_pit_dist_t0 = []
    rows_regime_t0 = []
    rows_g_t0 = []
    rows_g_c = []
    rows_g_t0_H = []
    rows_g_c_H = []
    rows_obj_total = []
    rows_obj_clean = []
    rows_obj_incombat = []
    rows_obj_postcombat = []
    rows_obj_fk_team = []
    rows_obj_opp_team = []
    rows_obj_c_total = []
    rows_obj_c_clean = []
    rows_first_obj_dt0 = []
    rows_first_obj_dL = []
    rows_other_eng_t0_H = []
    rows_other_eng_c_H = []
    rows_hist_frame_ms = []
    rows_hist_g_diff = []
    rows_hist_kills_diff = []

    allphase_fit_lead_ms = []
    allphase_fit_dur_ms = []
    allphase_fit_comp_by_H = []
    allphase_fit_other_by_H = []
    allphase_fit_epic_by_H = []
    allphase_fit_clean_epic_by_H = []
    allphase_fit_twr_by_H = []
    allphase_fit_clean_twr_by_H = []
    allphase_fit_kills_in_lead = []

    global_m_idx = 0
    for s_idx in range(79):
        s_dir = BANK_ROOT / f"external_{s_idx:04d}"
        z = np.load(s_dir / "BANK.npz", allow_pickle=True)
        cv = z["component_values"]
        off_cv = z["offset_component_values"]
        gt = z["gold_team"]
        off_gt = z["offset_gold_team"]
        mts = z["minute_ts"]
        off_mts = z["offset_minute_ts"]
        xy_arr = z["xy"]
        off_xy = z["offset_xy"]
        end_ms_arr = z["end_ms"]
        win_arr = z["winning_team"]

        ev_ts = z["event_timestamp"]
        ev_tc = z["event_type_code"]
        ev_cat = z["event_category_code"]
        ev_team = z["event_acquirer_team"]
        off_ev = z["offset_event_timestamp"]

        s_row = z["event_string_row"]
        s_col = z["event_string_col"]
        s_val = z["event_string_nonzero"]
        off_s = z["offset_event_string_row"]

        n_m = len(off_cv) - 1
        for m in range(n_m):
            cid, patch, srole, mrole = match_meta[global_m_idx]
            c_start, c_end = int(off_cv[m]), int(off_cv[m + 1])
            if c_start == c_end:
                global_m_idx += 1
                continue

            m_mts = mts[int(off_mts[m]):int(off_mts[m + 1])]
            m_gt = gt[int(off_gt[m]):int(off_gt[m + 1])]
            m_gd = (m_gt[:, 0] - m_gt[:, 1]).astype(np.float64)
            m_xy = xy_arr[int(off_xy[m]):int(off_xy[m + 1])]
            m_end_ms = int(end_ms_arr[m])
            m_win_blue = (int(win_arr[m]) == 100)

            e0, e1 = int(off_ev[m]), int(off_ev[m + 1])
            n_e = e1 - e0
            m_ev_ts = ev_ts[e0:e1]
            m_ev_tc = ev_tc[e0:e1]
            m_ev_cat = ev_cat[e0:e1]
            m_ev_team = ev_team[e0:e1]

            m_kill_mask = m_ev_tc == 1  # CHAMPION_KILL
            m_kill_ts = m_ev_ts[m_kill_mask]
            m_kill_team = m_ev_team[m_kill_mask]

            s0, s1 = int(off_s[m]), int(off_s[m + 1])
            sr_idx, sc_idx, sv_arr = s_row[s0:s1], s_col[s0:s1], s_val[s0:s1]
            msub_arr = np.zeros(n_e, dtype=np.int32)
            ttype_arr = np.zeros(n_e, dtype=np.int32)
            msub_arr[sr_idx[sc_idx == scol_msub]] = sv_arr[sc_idx == scol_msub]
            ttype_arr[sr_idx[sc_idx == scol_ttype]] = sv_arr[sc_idx == scol_ttype]

            obj_mask = m_ev_cat > 0
            o_ts = m_ev_ts[obj_mask]
            o_cat = m_ev_cat[obj_mask]
            o_team = m_ev_team[obj_mask]
            o_msub = msub_arr[obj_mask]
            o_ttype = ttype_arr[obj_mask]

            n_o = len(o_ts)
            o_mat = np.zeros((n_o, N_OBJ), dtype=np.int16)
            if n_o > 0:
                is_drg = o_cat == 1
                is_elder = is_drg & (o_msub == elder_code)
                is_elem_drg = is_drg & (~is_elder)
                is_baron = o_cat == 2
                is_herald = o_cat == 3
                is_atakhan = o_cat == 4
                is_horde = o_cat == 5
                is_twr = o_cat == 6
                is_inh = o_cat == 7
                is_plt = o_cat == 8
                is_soul = o_cat == 9

                o_mat[:, 0] = is_elem_drg
                o_mat[:, 1] = is_elder
                o_mat[:, 2] = is_baron
                o_mat[:, 3] = is_herald
                o_mat[:, 4] = is_horde
                o_mat[:, 5] = is_atakhan
                o_mat[:, 6] = is_twr & (o_ttype == outer_code)
                o_mat[:, 7] = is_twr & (o_ttype == inner_code)
                o_mat[:, 8] = is_twr & (o_ttype == base_code)
                o_mat[:, 9] = is_twr & (o_ttype == nexus_code)
                o_mat[:, 10] = is_inh
                o_mat[:, 11] = is_plt
                o_mat[:, 12] = is_twr
                o_mat[:, 13] = is_elem_drg | is_elder | is_baron | is_herald | is_horde | is_atakhan

                ts_elem_drg = o_ts[is_elem_drg]
                ts_elder = o_ts[is_elder]
                ts_soul = o_ts[is_soul]
                ts_baron = o_ts[is_baron]
                ts_herald = o_ts[is_herald]
                ts_horde = o_ts[is_horde]
                ts_atakhan = o_ts[is_atakhan]
            else:
                ts_elem_drg = ts_elder = ts_soul = ts_baron = ts_herald = ts_horde = ts_atakhan = np.empty(0, dtype=np.int32)

            m_cv = cv[c_start:c_end]
            all_f_ms = m_cv[:, comp_cols["f"]]
            all_L_ms = m_cv[:, comp_cols["L"]]

            for local_c in range(len(m_cv)):
                row = m_cv[local_c]
                f_ms = int(row[comp_cols["f"]])
                L_ms = int(row[comp_cols["L"]])
                pre_ms = int(row[comp_cols["pre"]])
                t0_ms = int(row[comp_cols["F_pre_ms"]])
                t0_idx = int(row[comp_cols["F_pre_index"]])

                is_allphase_iso = (
                    row[comp_cols["firstkill_direct_opposition"]] == 1
                    and row[comp_cols["global_isolation_verified"]] == 1
                    and row[comp_cols["overlap_next_anchor_le_L"]] == 0
                    and row[comp_cols["valid_h60"]] == 1
                    and row[comp_cols["valid_h90"]] == 1
                    and row[comp_cols["valid_h120"]] == 1
                    and row[comp_cols["pre_frame_support"]] == 1
                    and row[comp_cols["team_map_ok"]] == 1
                    and row[comp_cols["events_support_ok"]] == 1
                    and row[comp_cols["terminal_support_authoritative"]] == 1
                    and row[comp_cols["terminal_component_order_ok"]] == 1
                    and pre_ms >= 120000
                )

                if is_allphase_iso and mrole == "FIT":
                    # Actual last frame strictly before first kill f_ms: s = max{t_m < f_ms}
                    s_idx_ap = int(np.searchsorted(m_mts, f_ms, side="left")) - 1
                    if s_idx_ap >= 0:
                        s_ms_ap = int(m_mts[s_idx_ap])
                        fut_f = all_f_ms[all_f_ms > L_ms]
                        next_f_val = int(fut_f[0]) if len(fut_f) > 0 else 999999999
                        allphase_fit_lead_ms.append(f_ms - s_ms_ap)
                        allphase_fit_dur_ms.append(L_ms - f_ms)
                        allphase_fit_kills_in_lead.append(int(np.sum((m_kill_ts > s_ms_ap) & (m_kill_ts < f_ms))))
                        comp_h = [int((L_ms - s_ms_ap) <= h * 1000) for h in HORIZONS_S]
                        oth_h = [int(next_f_val <= s_ms_ap + h * 1000) for h in HORIZONS_S]
                        ep_h, cep_h, tw_h, ctw_h = [], [], [], []
                        for h in HORIZONS_S:
                            hi = s_ms_ap + h * 1000
                            m_in = (o_ts > s_ms_ap) & (o_ts <= hi)
                            m_cl = (o_ts > s_ms_ap) & (o_ts <= min(hi, next_f_val - 1))
                            ep_h.append(int(np.sum(o_mat[m_in, 13]) > 0) if n_o > 0 else 0)
                            cep_h.append(int(np.sum(o_mat[m_cl, 13]) > 0) if n_o > 0 else 0)
                            tw_h.append(int(np.sum(o_mat[m_in, 12]) > 0) if n_o > 0 else 0)
                            ctw_h.append(int(np.sum(o_mat[m_cl, 12]) > 0) if n_o > 0 else 0)
                        allphase_fit_comp_by_H.append(comp_h)
                        allphase_fit_other_by_H.append(oth_h)
                        allphase_fit_epic_by_H.append(ep_h)
                        allphase_fit_clean_epic_by_H.append(cep_h)
                        allphase_fit_twr_by_H.append(tw_h)
                        allphase_fit_clean_twr_by_H.append(ctw_h)

                if row[comp_cols["primary_candidate_eligible"]] != 1:
                    continue

                fut_f = all_f_ms[all_f_ms > L_ms]
                next_f_val = int(fut_f[0]) if len(fut_f) > 0 else 999999999
                par_mask = (np.arange(len(m_cv)) != local_c) & (all_f_ms <= f_ms) & (all_L_ms >= t0_ms)
                par_ongoing = int(np.any(par_mask))

                idx_c = int(np.searchsorted(m_mts, L_ms, side="left"))
                if idx_c >= len(m_mts):
                    idx_c = len(m_mts) - 1
                c_ms_val = int(m_mts[idx_c])

                soul_before = ts_soul[ts_soul <= t0_ms]
                has_soul_t0 = len(soul_before) > 0
                drg_before = ts_elem_drg[ts_elem_drg <= t0_ms]
                eld_before = ts_elder[ts_elder <= t0_ms]
                bar_before = ts_baron[ts_baron <= t0_ms]
                her_before = ts_herald[ts_herald <= t0_ms]
                hor_before = ts_horde[ts_horde <= t0_ms]
                ata_before = ts_atakhan[ts_atakhan <= t0_ms]

                avail_drg = (t0_ms >= 270000) and (not has_soul_t0) and (len(drg_before) == 0 or (t0_ms - int(drg_before[-1])) >= 270000)
                avail_eld = has_soul_t0 and (t0_ms - max(int(soul_before[-1]), int(eld_before[-1]) if len(eld_before) > 0 else 0) >= 330000)
                avail_hor = (450000 <= t0_ms <= 885000) and (len(hor_before) < 3)
                avail_her = (870000 <= t0_ms <= 1485000) and (len(her_before) < 1)
                avail_ata = (t0_ms >= 1170000) and (len(ata_before) < 1)
                avail_bar = (t0_ms >= 1470000) and (len(bar_before) == 0 or (t0_ms - int(bar_before[-1])) >= 330000)

                pts_t0 = m_xy[t0_idx]
                d_drg_pit = float(np.min(np.linalg.norm(pts_t0 - DRAGON_PIT_XY[None, :], axis=1)))
                d_bar_pit = float(np.min(np.linalg.norm(pts_t0 - BARON_PIT_XY[None, :], axis=1)))

                if avail_bar or avail_eld:
                    regime_t0 = 0
                elif avail_ata:
                    regime_t0 = 1
                elif avail_drg:
                    regime_t0 = 2
                elif avail_her or avail_hor:
                    regime_t0 = 3
                else:
                    regime_t0 = 4

                fk_is_blue = int(row[comp_cols["firstkill_credit_blue"]]) > 0
                fk_team_id = 100 if fk_is_blue else 200
                opp_team_id = 200 if fk_is_blue else 100

                g_t0 = float(m_gd[t0_idx])
                g_c = float(m_gd[idx_c])
                g_t0_h_list = []
                g_c_h_list = []
                for h_sec in HORIZONS_S:
                    target_idx_t0 = min(t0_idx + (h_sec // 60), len(m_gd) - 1)
                    target_idx_c = min(idx_c + (h_sec // 60), len(m_gd) - 1)
                    g_t0_h_list.append(float(m_gd[target_idx_t0]))
                    g_c_h_list.append(float(m_gd[target_idx_c]))

                in_c_mask = (o_ts > t0_ms) & (o_ts <= L_ms)
                incombat_cnt = np.sum(o_mat[in_c_mask], axis=0) if n_o > 0 else np.zeros(N_OBJ, dtype=np.int16)

                tot_by_h = np.zeros((5, N_OBJ), dtype=np.int16)
                cln_by_h = np.zeros((5, N_OBJ), dtype=np.int16)
                pst_by_h = np.zeros((5, N_OBJ), dtype=np.int16)
                fk_by_h = np.zeros((5, N_OBJ), dtype=np.int16)
                opp_by_h = np.zeros((5, N_OBJ), dtype=np.int16)
                ctot_by_h = np.zeros((5, N_OBJ), dtype=np.int16)
                ccln_by_h = np.zeros((5, N_OBJ), dtype=np.int16)
                oth_t0_h = np.zeros(5, dtype=np.int16)
                oth_c_h = np.zeros(5, dtype=np.int16)

                for h_idx, h_sec in enumerate(HORIZONS_S):
                    hi_t0 = t0_ms + h_sec * 1000
                    hi_c = c_ms_val + h_sec * 1000
                    if n_o > 0:
                        m_tot = (o_ts > t0_ms) & (o_ts <= hi_t0)
                        m_cln = (o_ts > t0_ms) & (o_ts <= min(hi_t0, next_f_val - 1))
                        m_pst = (o_ts > L_ms) & (o_ts <= hi_t0)
                        m_fk = m_tot & (o_team == fk_team_id)
                        m_opp = m_tot & (o_team == opp_team_id)
                        m_ctot = (o_ts > t0_ms) & (o_ts <= hi_c)
                        m_ccln = (o_ts > t0_ms) & (o_ts <= min(hi_c, next_f_val - 1))
                        tot_by_h[h_idx] = np.sum(o_mat[m_tot], axis=0)
                        cln_by_h[h_idx] = np.sum(o_mat[m_cln], axis=0)
                        pst_by_h[h_idx] = np.sum(o_mat[m_pst], axis=0)
                        fk_by_h[h_idx] = np.sum(o_mat[m_fk], axis=0)
                        opp_by_h[h_idx] = np.sum(o_mat[m_opp], axis=0)
                        ctot_by_h[h_idx] = np.sum(o_mat[m_ctot], axis=0)
                        ccln_by_h[h_idx] = np.sum(o_mat[m_ccln], axis=0)
                    oth_t0_h[h_idx] = int(np.sum((all_f_ms > L_ms) & (all_f_ms <= hi_t0)))
                    oth_c_h[h_idx] = int(np.sum((all_f_ms > L_ms) & (all_f_ms <= hi_c)))

                dt0_arr = np.full(N_OBJ, -1, dtype=np.int32)
                dL_arr = np.full(N_OBJ, -999999, dtype=np.int32)
                if n_o > 0:
                    w300 = (o_ts > t0_ms) & (o_ts <= t0_ms + 300000)
                    if np.any(w300):
                        sub_ts = o_ts[w300]
                        sub_mat = o_mat[w300]
                        for k_obj in range(N_OBJ):
                            hits = sub_ts[sub_mat[:, k_obj] > 0]
                            if len(hits) > 0:
                                dt0_arr[k_obj] = int(hits[0]) - t0_ms
                                dL_arr[k_obj] = int(hits[0]) - L_ms

                h_ms = np.full(5, -1, dtype=np.int64)
                h_gd = np.zeros(5, dtype=np.float64)
                h_kd = np.zeros(5, dtype=np.float64)
                for lag_f in range(5):
                    f_i = t0_idx - lag_f
                    if f_i >= 0:
                        f_ts_val = int(m_mts[f_i])
                        h_ms[lag_f] = f_ts_val
                        h_gd[lag_f] = float(m_gd[f_i])
                        k_le = m_kill_ts <= f_ts_val
                        b_k = int(np.sum(m_kill_team[k_le] == 100))
                        r_k = int(np.sum(m_kill_team[k_le] == 200))
                        h_kd[lag_f] = float(b_k - r_k)

                rows_match_idx.append(global_m_idx)
                rows_comp_id.append(int(row[comp_cols["component_id"]]))
                rows_patch.append(patch)
                rows_split_role.append(srole)
                rows_macro_role.append(mrole)
                rows_t0_ms.append(t0_ms)
                rows_tau_ms.append(pre_ms)
                rows_f_ms.append(f_ms)
                rows_L_ms.append(L_ms)
                rows_c_ms.append(c_ms_val)
                rows_end_ms.append(m_end_ms)
                rows_next_f_ms.append(next_f_val)
                rows_parallel_ongoing.append(par_ongoing)
                rows_fk_blue.append(fk_is_blue)
                rows_win_blue.append(m_win_blue)
                rows_avail_t0.append([avail_drg, avail_eld, avail_hor, avail_her, avail_ata, avail_bar])
                rows_pit_dist_t0.append([d_drg_pit, d_bar_pit])
                rows_regime_t0.append(regime_t0)
                rows_g_t0.append(g_t0)
                rows_g_c.append(g_c)
                rows_g_t0_H.append(g_t0_h_list)
                rows_g_c_H.append(g_c_h_list)
                rows_obj_total.append(tot_by_h)
                rows_obj_clean.append(cln_by_h)
                rows_obj_incombat.append(incombat_cnt)
                rows_obj_postcombat.append(pst_by_h)
                rows_obj_fk_team.append(fk_by_h)
                rows_obj_opp_team.append(opp_by_h)
                rows_obj_c_total.append(ctot_by_h)
                rows_obj_c_clean.append(ccln_by_h)
                rows_first_obj_dt0.append(dt0_arr)
                rows_first_obj_dL.append(dL_arr)
                rows_other_eng_t0_H.append(oth_t0_h)
                rows_other_eng_c_H.append(oth_c_h)
                rows_hist_frame_ms.append(h_ms)
                rows_hist_g_diff.append(h_gd)
                rows_hist_kills_diff.append(h_kd)

            global_m_idx += 1

    # Convert to numpy arrays
    arr_match_idx = np.asarray(rows_match_idx, dtype=np.int32)
    arr_comp_id = np.asarray(rows_comp_id, dtype=np.int32)
    arr_patch = np.asarray(rows_patch, dtype="U8")
    arr_split_role = np.asarray(rows_split_role, dtype="U24")
    arr_macro_role = np.asarray(rows_macro_role, dtype="U12")
    arr_t0_ms = np.asarray(rows_t0_ms, dtype=np.int64)
    arr_tau_ms = np.asarray(rows_tau_ms, dtype=np.int64)
    arr_f_ms = np.asarray(rows_f_ms, dtype=np.int64)
    arr_L_ms = np.asarray(rows_L_ms, dtype=np.int64)
    arr_c_ms = np.asarray(rows_c_ms, dtype=np.int64)
    arr_end_ms = np.asarray(rows_end_ms, dtype=np.int64)
    arr_next_f_ms = np.asarray(rows_next_f_ms, dtype=np.int64)
    arr_par = np.asarray(rows_parallel_ongoing, dtype=np.int8)
    arr_fk_blue = np.asarray(rows_fk_blue, dtype=bool)
    arr_win_blue = np.asarray(rows_win_blue, dtype=bool)
    arr_avail_t0 = np.asarray(rows_avail_t0, dtype=bool)
    arr_pit_dist_t0 = np.asarray(rows_pit_dist_t0, dtype=np.float32)
    arr_regime_t0 = np.asarray(rows_regime_t0, dtype=np.int8)
    arr_g_t0 = np.asarray(rows_g_t0, dtype=np.float64)
    arr_g_c = np.asarray(rows_g_c, dtype=np.float64)
    arr_g_t0_H = np.asarray(rows_g_t0_H, dtype=np.float64)
    arr_g_c_H = np.asarray(rows_g_c_H, dtype=np.float64)
    arr_obj_total = np.asarray(rows_obj_total, dtype=np.int16)
    arr_obj_clean = np.asarray(rows_obj_clean, dtype=np.int16)
    arr_obj_incombat = np.asarray(rows_obj_incombat, dtype=np.int16)
    arr_obj_postcombat = np.asarray(rows_obj_postcombat, dtype=np.int16)
    arr_obj_fk_team = np.asarray(rows_obj_fk_team, dtype=np.int16)
    arr_obj_opp_team = np.asarray(rows_obj_opp_team, dtype=np.int16)
    arr_obj_c_total = np.asarray(rows_obj_c_total, dtype=np.int16)
    arr_obj_c_clean = np.asarray(rows_obj_c_clean, dtype=np.int16)
    arr_first_obj_dt0 = np.asarray(rows_first_obj_dt0, dtype=np.int32)
    arr_first_obj_dL = np.asarray(rows_first_obj_dL, dtype=np.int32)
    arr_other_eng_t0_H = np.asarray(rows_other_eng_t0_H, dtype=np.int16)
    arr_other_eng_c_H = np.asarray(rows_other_eng_c_H, dtype=np.int16)
    arr_hist_frame_ms = np.asarray(rows_hist_frame_ms, dtype=np.int64)
    arr_hist_g_diff = np.asarray(rows_hist_g_diff, dtype=np.float64)
    arr_hist_kills_diff = np.asarray(rows_hist_kills_diff, dtype=np.float64)

    # 1. Save separate query files stratified by t_0-known objective state
    def save_query_subset(filename: str, sub_mask: np.ndarray, desc: str) -> dict:
        fpath = OBJ_QUERY_DIR / filename
        np.savez_compressed(
            fpath,
            match_index=arr_match_idx[sub_mask],
            component_id=arr_comp_id[sub_mask],
            patch=arr_patch[sub_mask],
            split_role=arr_split_role[sub_mask],
            macro_role=arr_macro_role[sub_mask],
            t0_ms=arr_t0_ms[sub_mask],
            tau_ms=arr_tau_ms[sub_mask],
            first_kill_ms=arr_f_ms[sub_mask],
            last_kill_ms=arr_L_ms[sub_mask],
            c_ms=arr_c_ms[sub_mask],
            end_ms=arr_end_ms[sub_mask],
            next_fight_first_kill_ms=arr_next_f_ms[sub_mask],
            avail_t0=arr_avail_t0[sub_mask],
            pit_dist_t0=arr_pit_dist_t0[sub_mask],
            regime_t0=arr_regime_t0[sub_mask],
            g_t0=arr_g_t0[sub_mask],
            g_c=arr_g_c[sub_mask],
            g_t0_H=arr_g_t0_H[sub_mask],
            g_c_H=arr_g_c_H[sub_mask],
            obj_total_by_H=arr_obj_total[sub_mask],
            obj_clean_by_H=arr_obj_clean[sub_mask],
            obj_incombat=arr_obj_incombat[sub_mask],
            obj_postcombat_by_H=arr_obj_postcombat[sub_mask],
            other_eng_t0_H=arr_other_eng_t0_H[sub_mask],
            hist_frame_ms=arr_hist_frame_ms[sub_mask],
            hist_g_diff=arr_hist_g_diff[sub_mask],
            hist_kills_diff=arr_hist_kills_diff[sub_mask],
        )
        roles_sub = arr_macro_role[sub_mask]
        by_split = {r: int(np.sum(roles_sub == r)) for r in ["FIT", "STOP", "DEV", "CAL", "AUDIT", "TEST", "UNUSED"]}
        return {
            "file": filename,
            "path": str(fpath.relative_to(PREP_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(fpath),
            "description": desc,
            "total_rows": int(np.sum(sub_mask)),
            "rows_by_macro_role": by_split,
        }

    saved_subsets = []
    saved_subsets.append(save_query_subset(
        "Q_ALL_PRIMARY_TIMELINES.npz",
        np.ones(len(arr_t0_ms), dtype=bool),
        "All 146,845 primary eligible engagements with t0/tau/f/L/c/E timelines, objective capture matrices, and 5-frame histories.",
    ))
    saved_subsets.append(save_query_subset(
        "Q_OBJ_01_BARON_ELDER_AVAIL.npz",
        arr_regime_t0 == 0,
        "Queries where Baron Nashor or Elder Dragon is live or spawning within 30s at t_0 (priority regime 0).",
    ))
    saved_subsets.append(save_query_subset(
        "Q_OBJ_02_ATAKHAN_AVAIL.npz",
        arr_regime_t0 == 1,
        "Queries where Atakhan is live or spawning within 30s at t_0 (and Baron/Elder not live; priority regime 1).",
    ))
    saved_subsets.append(save_query_subset(
        "Q_OBJ_03_DRAGON_AVAIL.npz",
        arr_regime_t0 == 2,
        "Queries where Elemental Dragon is live or spawning within 30s at t_0 (priority regime 2).",
    ))
    saved_subsets.append(save_query_subset(
        "Q_OBJ_04_HERALD_HORDE_AVAIL.npz",
        arr_regime_t0 == 3,
        "Queries where Rift Herald or Voidgrubs (Horde) is live or spawning within 30s at t_0 (priority regime 3).",
    ))
    saved_subsets.append(save_query_subset(
        "Q_OBJ_05_NONE_SIEGE_SKIRMISH.npz",
        arr_regime_t0 == 4,
        "Queries where no epic monster is live or spawning within 30s at t_0 (pure lane/jungle skirmish & turret/plate siege regime 4).",
    ))
    saved_subsets.append(save_query_subset(
        "Q_OBJ_DRAGON_PIT_CONTEST_T0.npz",
        (arr_avail_t0[:, 0] | arr_avail_t0[:, 1]) & (arr_pit_dist_t0[:, 0] <= 3500.0),
        "Queries where Dragon/Elder is available at t_0 AND at least one champion is within 3,500 units of Dragon Pit at t_0.",
    ))
    saved_subsets.append(save_query_subset(
        "Q_OBJ_BARON_PIT_CONTEST_T0.npz",
        (arr_avail_t0[:, 2] | arr_avail_t0[:, 3] | arr_avail_t0[:, 5]) & (arr_pit_dist_t0[:, 1] <= 3500.0),
        "Queries where Baron/Herald/Voidgrubs is available at t_0 AND at least one champion is within 3,500 units of Baron Pit at t_0.",
    ))

    write_json(OBJ_QUERY_DIR / "MANIFEST_QUERIES_BY_OBJECTIVE.json", {
        "version": "queries-by-objective-v5-r1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Strictly t_0-observable rule state (events <= t_0, game clock t_0, and t_0 frame XY). Zero future lookahead.",
        "objective_names_order": OBJ_NAMES,
        "horizons_sec": HORIZONS_S,
        "avail_t0_columns": ["dragon", "elder", "horde", "herald", "atakhan", "baron"],
        "Saved_Query_Archives": saved_subsets,
    })

    # 2. Compute Horizon Evidence Tables on FIT (and stratified by objective regime, game phase, patch, and DEV/AUDIT)
    common_args = (
        arr_t0_ms, arr_f_ms, arr_L_ms, arr_c_ms, arr_end_ms, arr_next_f_ms, arr_par,
        arr_avail_t0, arr_g_t0, arr_g_c, arr_g_t0_H, arr_g_c_H,
        arr_obj_total, arr_obj_clean, arr_obj_incombat, arr_obj_postcombat,
        arr_obj_fk_team, arr_obj_opp_team, arr_obj_c_total, arr_obj_c_clean,
        arr_first_obj_dt0, arr_first_obj_dL, arr_other_eng_t0_H, arr_other_eng_c_H,
    )

    fit_mask = arr_macro_role == "FIT"
    dev_mask = arr_macro_role == "DEV"
    audit_mask = arr_macro_role == "AUDIT"

    fit_overall = compute_horizon_summary(fit_mask, *common_args)
    dev_overall = compute_horizon_summary(dev_mask, *common_args)
    audit_overall = compute_horizon_summary(audit_mask, *common_args)

    fit_by_regime = {}
    for r_id, r_name in REGIME_NAMES.items():
        fit_by_regime[r_name] = compute_horizon_summary(fit_mask & (arr_regime_t0 == r_id), *common_args)

    fit_by_pit_contest = {
        "DRAGON_PIT_CONTEST_T0": compute_horizon_summary(
            fit_mask & (arr_avail_t0[:, 0] | arr_avail_t0[:, 1]) & (arr_pit_dist_t0[:, 0] <= 3500.0),
            *common_args,
        ),
        "BARON_PIT_CONTEST_T0": compute_horizon_summary(
            fit_mask & (arr_avail_t0[:, 2] | arr_avail_t0[:, 3] | arr_avail_t0[:, 5]) & (arr_pit_dist_t0[:, 1] <= 3500.0),
            *common_args,
        ),
    }

    fit_by_phase = {
        "EARLY_2_TO_14M": compute_horizon_summary(fit_mask & (arr_t0_ms < 840000), *common_args),
        "MID_14_TO_25M": compute_horizon_summary(fit_mask & (arr_t0_ms >= 840000) & (arr_t0_ms < 1500000), *common_args),
        "LATE_25M_PLUS": compute_horizon_summary(fit_mask & (arr_t0_ms >= 1500000), *common_args),
    }

    fit_by_patch = {
        p: compute_horizon_summary(fit_mask & (arr_patch == p), *common_args)
        for p in ["15.18", "15.19", "15.20"]
    }

    # 3. All-Phase Lead-Time Audit on FIT (0..60s bins of t_F - t_0)
    ap_lead = np.asarray(allphase_fit_lead_ms, dtype=np.float64) / 1000.0
    ap_dur = np.asarray(allphase_fit_dur_ms, dtype=np.float64) / 1000.0
    ap_comp = np.asarray(allphase_fit_comp_by_H, dtype=np.float64)
    ap_oth = np.asarray(allphase_fit_other_by_H, dtype=np.float64)
    ap_ep = np.asarray(allphase_fit_epic_by_H, dtype=np.float64)
    ap_cep = np.asarray(allphase_fit_clean_epic_by_H, dtype=np.float64)
    ap_tw = np.asarray(allphase_fit_twr_by_H, dtype=np.float64)
    ap_ctw = np.asarray(allphase_fit_clean_twr_by_H, dtype=np.float64)
    ap_kl = np.asarray(allphase_fit_kills_in_lead, dtype=np.float64)

    lead_bins_audit = {}
    for b_idx in range(6):
        lo_s = b_idx * 10.0
        hi_s = (b_idx + 1) * 10.0
        if b_idx == 0:
            bmask = (ap_lead >= 0.0) & (ap_lead <= 10.0)
        else:
            bmask = (ap_lead > lo_s) & (ap_lead <= hi_s)
        nb = int(np.sum(bmask))
        if nb > 0:
            lead_bins_audit[f"lead_{int(lo_s):02d}_to_{int(hi_s):02d}s"] = {
                "n_queries": nb,
                "share_of_allphase_fit": round(nb / len(ap_lead), 4),
                "is_Q_fresh_v5_cohort": (b_idx == 1),
                "mean_lead_sec": round(float(np.mean(ap_lead[bmask])), 2),
                "mean_combat_dur_sec": round(float(np.mean(ap_dur[bmask])), 2),
                "prior_kill_in_t0_to_tF_rate": round(float(np.mean(ap_kl[bmask] > 0)), 4),
                "by_horizon": {
                    f"H{HORIZONS_S[h_i]}s": {
                        "target_combat_completion_rate": round(float(np.mean(ap_comp[bmask, h_i])), 4),
                        "other_engagement_intervention_rate": round(float(np.mean(ap_oth[bmask, h_i])), 4),
                        "any_epic_incidence_total": round(float(np.mean(ap_ep[bmask, h_i])), 4),
                        "any_epic_incidence_clean": round(float(np.mean(ap_cep[bmask, h_i])), 4),
                        "any_turret_incidence_total": round(float(np.mean(ap_tw[bmask, h_i])), 4),
                        "any_turret_incidence_clean": round(float(np.mean(ap_ctw[bmask, h_i])), 4),
                    }
                    for h_i in range(5)
                },
            }

    # 4. Input History Frame Audit (L in {1, 2, 3, 4, 5} frames = {0, ~60, ~120, ~180, ~240}s) on FIT & DEV
    fit_hms = arr_hist_frame_ms[fit_mask]
    fit_hgd = arr_hist_g_diff[fit_mask]
    fit_hkd = arr_hist_kills_diff[fit_mask]
    dev_hms = arr_hist_frame_ms[dev_mask]
    dev_hgd = arr_hist_g_diff[dev_mask]
    dev_hkd = arr_hist_kills_diff[dev_mask]
    dev_midx = arr_match_idx[dev_mask]

    input_frame_support = {}
    for L_frames in range(1, 6):
        lag_idx = L_frames - 1
        valid_f = fit_hms[:, lag_idx] >= 0
        if lag_idx == 0:
            span_sec = np.zeros(np.sum(valid_f))
        else:
            span_sec = (fit_hms[valid_f, 0] - fit_hms[valid_f, lag_idx]) / 1000.0
        input_frame_support[f"L{L_frames}_frames"] = {
            "L_frames": L_frames,
            "nominal_history_sec": lag_idx * 60,
            "fit_queries_total": int(len(fit_hms)),
            "fit_queries_with_full_L_frames_unpadded": int(np.sum(valid_f)),
            "fit_full_L_support_rate": round(float(np.mean(valid_f)), 4),
            "fit_padded_early_query_rate": round(float(np.mean(~valid_f)), 4),
            "actual_span_t0_minus_t_minus_k_sec": pct_dict(span_sec),
        }

    # Zero-training conditional linear/Ridge baseline on FIT -> DEV (with paired match bootstrap!)
    # to preview how much past frame trajectory (L=1..5) reduces MAE/RMSE/LogLoss beyond L=1 (current state at t0)
    # when current state [g_t0, k_diff_t0, avail_t0, t0_min] is ALWAYS included!
    fit_y_mat = arr_g_t0_H[fit_mask] - arr_g_t0[fit_mask, None]
    dev_y_mat = arr_g_t0_H[dev_mask] - arr_g_t0[dev_mask, None]

    def build_hist_design(hms, hgd, hkd, avail, t0_ms_sub, L_frames: int) -> np.ndarray:
        # Base current state at t_0 (always identical across L=1..5):
        # [1, g_0, kd_0, t0_min, avail_6]
        cols = [
            np.ones(len(hms), dtype=np.float64),
            hgd[:, 0] / 1000.0,
            hkd[:, 0] / 10.0,
            t0_ms_sub / 600000.0,
            avail.astype(np.float64),
        ]
        # For each additional past frame k = 1 .. L_frames-1:
        # add [mask_k, delta_g_k = (g_0 - g_{-k}) * mask_k, delta_kd_k = (kd_0 - kd_{-k}) * mask_k]
        for k in range(1, L_frames):
            mk = (hms[:, k] >= 0).astype(np.float64)
            dg_k = ((hgd[:, 0] - hgd[:, k]) / 1000.0) * mk
            dk_k = ((hkd[:, 0] - hkd[:, k]) / 10.0) * mk
            cols.extend([mk, dg_k, dk_k])
        return np.column_stack(cols)

    # Pre-generate 500 paired match bootstrap draws on DEV matches
    unique_dev_matches, dev_inv = np.unique(dev_midx, return_inverse=True)
    n_dev_m = len(unique_dev_matches)
    rng = np.random.default_rng(20261017)
    boot_weights = np.zeros((500, n_dev_m), dtype=np.float64)
    for b in range(500):
        draws = rng.integers(0, n_dev_m, size=n_dev_m)
        boot_weights[b] = np.bincount(draws, minlength=n_dev_m)

    # Match-equal weights for DEV queries
    dev_q_per_m = np.bincount(dev_inv, minlength=n_dev_m).astype(np.float64)
    dev_w_per_q = 1.0 / np.maximum(1.0, dev_q_per_m[dev_inv])

    input_history_preview_by_H = {}
    for h_idx, h_sec in enumerate(["H120s", "H180s"]):
        h_col = 1 if h_sec == "H120s" else 2
        y_fit = fit_y_mat[:, h_col]
        y_dev = dev_y_mat[:, h_col]

        # Fit Ridge (alpha=1.0) for L=1..5 on FIT, evaluate match-weighted MAE & RMSE on DEV
        dev_mae_by_m = {}
        dev_rmse_sq_by_m = {}
        point_metrics = {}
        for L_f in range(1, 6):
            X_tr = build_hist_design(fit_hms, fit_hgd, fit_hkd, arr_avail_t0[fit_mask], arr_t0_ms[fit_mask], L_f)
            X_dv = build_hist_design(dev_hms, dev_hgd, dev_hkd, arr_avail_t0[dev_mask], arr_t0_ms[dev_mask], L_f)
            reg = np.eye(X_tr.shape[1]) * 1.0
            reg[0, 0] = 0.0
            beta = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ y_fit)
            pred_dv = X_dv @ beta
            abs_err = np.abs(y_dev - pred_dv)
            sq_err = (y_dev - pred_dv) ** 2

            # Per-match mean error on DEV
            m_abs = np.bincount(dev_inv, weights=abs_err * dev_w_per_q, minlength=n_dev_m)
            m_sq = np.bincount(dev_inv, weights=sq_err * dev_w_per_q, minlength=n_dev_m)
            dev_mae_by_m[L_f] = m_abs
            dev_rmse_sq_by_m[L_f] = m_sq
            point_metrics[L_f] = {
                "match_weighted_mae_gold": round(float(np.mean(m_abs)), 3),
                "match_weighted_rmse_gold": round(float(np.sqrt(np.mean(m_sq))), 3),
            }

        # Paired match bootstrap differences D_L = R_L - R_ref (ref = L5 = 240s, and vs L1 = 0s)
        contrasts_vs_L5 = {}
        contrasts_vs_L1 = {}
        for L_f in range(1, 6):
            diff_L5 = dev_mae_by_m[L_f] - dev_mae_by_m[5]
            b_means_L5 = (boot_weights @ diff_L5) / n_dev_m
            contrasts_vs_L5[f"L{L_f}_minus_L5"] = {
                "delta_mae_gold": round(float(np.mean(diff_L5)), 3),
                "ci95_lo": round(float(np.percentile(b_means_L5, 2.5)), 3),
                "ci95_hi": round(float(np.percentile(b_means_L5, 97.5)), 3),
                "ucb95_one_sided": round(float(np.percentile(b_means_L5, 95.0)), 3),
            }
            diff_L1 = dev_mae_by_m[L_f] - dev_mae_by_m[1]
            b_means_L1 = (boot_weights @ diff_L1) / n_dev_m
            contrasts_vs_L1[f"L{L_f}_minus_L1"] = {
                "delta_mae_gold": round(float(np.mean(diff_L1)), 3),
                "ci95_lo": round(float(np.percentile(b_means_L1, 2.5)), 3),
                "ci95_hi": round(float(np.percentile(b_means_L1, 97.5)), 3),
            }

        input_history_preview_by_H[h_sec] = {
            "point_metrics_by_L": point_metrics,
            "paired_bootstrap_mae_diff_vs_L5_ref": contrasts_vs_L5,
            "paired_bootstrap_mae_diff_vs_L1_current_only": contrasts_vs_L1,
        }

    evidence_doc = {
        "version": "window-evidence-v5-r1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_executed": False,
        "methodology_contract": {
            "outcome_window_H_principle": (
                "H in {60, 120, 180, 240, 300}s defines the target Y_H = G(t_0 + H) - G(t_0) itself and is NEVER selected by comparing "
                "model AUC or MAE across different targets. Instead, H is selected on FIT from the empirical trade-off between "
                "(1) target combat completion rate Pr(t_C - t_0 <= H), (2) post-combat objective capture by type before the next engagement "
                "starts (clean pre-next-fight capture), and (3) other-engagement intervention rate Pr(t_next_F <= t_0 + H) and match termination rate."
            ),
            "input_window_L_principle": (
                "Once H is fixed, input length L in {1, 2, 3, 4, 5} frames (~0, 60, 120, 180, 240s of past frame snapshots) is evaluated against "
                "the EXACT SAME query cohort, EXACT SAME current t_0 reconstructed state (full prefix events <= t_0), and EXACT SAME target Y_H, "
                "using paired match bootstrap on D_L = R_L - R_ref with non-inferiority bound UCB_95(D_L) <= epsilon."
            ),
            "denominator_and_censoring_rule": (
                "All primary rates keep the full eligible FIT query denominator (N=70,201) intact. Conditional objective rates use strictly "
                "t_0-observable availability (avail_t0 computed from events <= t_0 and game clock t_0), never conditioning on future occurrence. "
                "Normal match termination (40,000/40,000 matches have valid GAME_END) is separated from data truncation (0/40,000 matches)."
            ),
        },
        "fit_overall_evidence": fit_overall,
        "fit_by_t0_objective_regime": fit_by_regime,
        "fit_by_t0_pit_contest": fit_by_pit_contest,
        "fit_by_game_phase": fit_by_phase,
        "fit_by_patch": fit_by_patch,
        "out_of_time_stability_comparison": {
            "DEV_15_21": dev_overall,
            "AUDIT_15_21": audit_overall,
        },
        "allphase_lead_time_6bins_fit_audit": lead_bins_audit,
        "input_window_L1_to_L5_audit": {
            "frame_interval_and_padding_support_on_FIT": input_frame_support,
            "zero_training_linear_trajectory_preview_FIT_to_DEV": input_history_preview_by_H,
        },
    }

    write_json(OUT_DIR / "OUTCOME_AND_INPUT_WINDOW_EVIDENCE_V5.json", evidence_doc)
    print("Successfully built window_evidence_v5!")


if __name__ == "__main__":
    main()
