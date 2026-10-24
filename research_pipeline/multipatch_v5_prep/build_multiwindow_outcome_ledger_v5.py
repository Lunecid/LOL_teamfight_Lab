"""Build Multi-Window Event -> Reward -> Outcome Ledger (v5) on kr_15_18_22.

Implements:
1. Event Table & Reward Provenance Decomposition (DIRECT_OBSERVED, RULE_EXACT,
   ESTIMATED_WITH_ASSUMPTION, UNVERIFIED_MISSING) without double counting.
2. Both Marginal Slices [(t0, t1], (t1, t2], (t2, t3]] (0-60s, 60-120s, 120-180s)
   and Cumulative Windows [(t0, t1], (t0, t2], (t0, t3]] (0-60s, 0-120s, 0-180s),
   preserving BOTH Blue and Red raw values.
3. End-of-Window Point-in-Time States (Dragon stacks, Soul, Baron/Elder active buffs,
   remaining Turrets by tier, active destroyed Inhibitors) kept distinct from additive counts.
4. Four Separate Outcome Records per Window:
   - Kill Exchange (Target combat & Window)
   - Economic Outcome (Blue dG, Red dG, net dG, Econ Winner)
   - Pareto Dominance Composite Verdict (BLUE_DOMINANT, RED_DOMINANT, CONFLICTED_EXCHANGE, NO_DIFFERENCE)
   - Match Termination Outcome (ended_within_H, game_winner) without overwriting Economic/Composite labels.
5. FIT Empirical Verification of Econ vs Objective/Structure agreement, Conflict subtypes,
   60s -> 120s -> 180s Transition Matrices (overall & by t_0 objective regime), and
   Multi-Target Input Window (L=1..5) Baseline Comparison across Gold, Objectives, Structures, and States.

Strictly zero deep model training.
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
LEDGER_DIR = OUT_DIR / "multiwindow_outcome_ledger"
OBJ_QUERY_DIR = OUT_DIR / "queries_by_objective"
EXT_D = Path(os.environ["LOL_EXTERNAL_ROOT"])
BANK_ROOT = EXT_D / "bank" / "kr_15_18_22"

REGIME_NAMES = {
    0: "BARON_OR_ELDER_AVAIL",
    1: "ATAKHAN_AVAIL",
    2: "DRAGON_AVAIL",
    3: "HERALD_OR_HORDE_AVAIL",
    4: "NONE_SIEGE_SKIRMISH",
}

VERDICT_NAMES = {
    0: "BLUE_DOMINANT",
    1: "RED_DOMINANT",
    2: "CONFLICTED_EXCHANGE",
    3: "NO_DIFFERENCE",
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


def pareto_verdict_from_diffs(diff_mat: np.ndarray) -> np.ndarray:
    """Given diff_mat of shape (N, K) where >0 favors Blue and <0 favors Red:
    Returns integer array of shape (N,):
      0: BLUE_DOMINANT (all >= 0 and at least one > 0)
      1: RED_DOMINANT  (all <= 0 and at least one < 0)
      2: CONFLICTED_EXCHANGE (at least one > 0 and at least one < 0)
      3: NO_DIFFERENCE (all == 0)
    """
    has_pos = np.any(diff_mat > 0, axis=1)
    has_neg = np.any(diff_mat < 0, axis=1)
    out = np.full(len(diff_mat), 3, dtype=np.int8)
    out[has_pos & (~has_neg)] = 0
    out[has_neg & (~has_pos)] = 1
    out[has_pos & has_neg] = 2
    return out


def main():
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    fm = json.loads((BANK_ROOT / "FIELD_MANIFEST.json").read_text(encoding="utf-8"))
    comp_cols = {name: idx for idx, name in enumerate(fm["component_columns"])}

    # Event type codes
    tc_building = fm["event_type_names"].index("BUILDING_KILL")
    tc_kill = fm["event_type_names"].index("CHAMPION_KILL")
    tc_soul = fm["event_type_names"].index("DRAGON_SOUL_GIVEN")
    tc_monster = fm["event_type_names"].index("ELITE_MONSTER_KILL")
    tc_item_buy = fm["event_type_names"].index("ITEM_PURCHASED")
    tc_item_sell = fm["event_type_names"].index("ITEM_SOLD")
    tc_item_undo = fm["event_type_names"].index("ITEM_UNDO")
    tc_level_up = fm["event_type_names"].index("LEVEL_UP")
    tc_plate = fm["event_type_names"].index("TURRET_PLATE_DESTROYED")
    tc_ward_kill = fm["event_type_names"].index("WARD_KILL")
    tc_ward_place = fm["event_type_names"].index("WARD_PLACED")

    # Numeric field columns
    col_pid = fm["event_numeric_field_order"].index("participantId")
    col_bty = fm["event_numeric_field_order"].index("bounty")
    col_sbty = fm["event_numeric_field_order"].index("shutdownBounty")
    col_gg = fm["event_numeric_field_order"].index("goldGain")

    # String field columns
    scol_msub = fm["event_string_field_order"].index("monsterSubType")
    scol_ttype = fm["event_string_field_order"].index("towerType")
    scol_ltype = fm["event_string_field_order"].index("laneType")
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
    for be in bank_entries:
        mid = be["match_id"]
        patch = be["patch"]
        cid = mid.upper() if "_" in mid else f"KR_{mid}"
        bkt = bucket_from_key(salt_split, cid)
        srole, mrole = get_split_role(patch, bkt)
        match_meta.append((cid, patch, srole, mrole))

    # Load existing Q_ALL_PRIMARY_TIMELINES.npz for regime_t0 and hist arrays
    q_base = np.load(OBJ_QUERY_DIR / "Q_ALL_PRIMARY_TIMELINES.npz")
    N_Q = len(q_base["t0_ms"])
    print(f"Building multi-window event/reward/outcome ledger for {N_Q} primary queries...")

    # Preallocate arrays for the 3 Marginal Slices s in {0:(t0,t1], 1:(t1,t2], 2:(t2,t3]}
    # and 2 Teams t in {0:Blue(100), 1:Red(200)}
    # 1. Observed Frame Timestamps [t0, t1, t2, t3] and Raw Team Total Gold [G_Blue, G_Red] at [t0, t1, t2, t3]
    frame_ts_4 = np.zeros((N_Q, 4), dtype=np.int64)
    team_gold_4 = np.zeros((N_Q, 4, 2), dtype=np.float64)  # (N_Q, 4_frames, 2_teams)

    # 2. Target combat kill exchange in (t0, L_ms]: [kills, deaths, assists, direct_kill_gold, est_assist_gold] per team
    combat_kill_exch = np.zeros((N_Q, 2, 5), dtype=np.float64)

    # 3. Per-Slice (3 slices: 0-60s, 60-120s, 120-180s) x Per-Team (2: Blue, Red) Event Counts (13 event types):
    #   0: kills, 1: assists,
    #   2: elem_dragon, 3: elder_dragon, 4: baron, 5: atakhan, 6: herald, 7: horde,
    #   8: outer_turret, 9: inner_turret, 10: base_turret, 11: nexus_turret, 12: inhibitor, 13: plate,
    #   14: item_purchased, 15: item_sold, 16: item_undo, 17: ward_placed, 18: ward_kill, 19: level_up
    SLICE_EVENT_NAMES = [
        "kills", "assists",
        "elem_dragon", "elder_dragon", "baron", "atakhan", "herald", "horde",
        "outer_turret", "inner_turret", "base_turret", "nexus_turret", "inhibitor", "plate",
        "item_purchased", "item_sold", "item_undo", "ward_placed", "ward_kill", "level_up",
    ]
    N_SEV = len(SLICE_EVENT_NAMES)
    slice_team_events = np.zeros((N_Q, 3, 2, N_SEV), dtype=np.int16)

    # 4. Per-Slice (3) x Per-Team (2) Gold Reward Decomposition (8 components):
    #   0: champ_kill_direct_obs_gold (bounty + shutdownBounty, DIRECT_OBSERVED)
    #   1: champ_assist_est_gold (50% pool rule, ESTIMATED_WITH_ASSUMPTION)
    #   2: obj_bounty_direct_obs_gold (raw bounty on monster kill, DIRECT_OBSERVED)
    #   3: obj_base_rule_exact_gold (dragon 25 / elder 850 / baron 1525 / herald 100 / horde 95, RULE_EXACT)
    #   4: obj_local_est_gold (herald 200 / atakhan 100, ESTIMATED_WITH_ASSUMPTION)
    #   5: struct_bounty_direct_obs_gold (raw bounty on building kill, DIRECT_OBSERVED)
    #   6: struct_global_and_plate_rule_exact_gold (global turret 250/625 + inhib 50 + plate 125, RULE_EXACT)
    #   7: struct_local_est_gold (local turret proximity 250/425/375 + first turret 150, ESTIMATED_WITH_ASSUMPTION)
    GOLD_COMP_NAMES = [
        "champ_kill_direct_obs_gold",
        "champ_assist_est_gold",
        "obj_bounty_direct_obs_gold",
        "obj_base_rule_exact_gold",
        "obj_local_est_gold",
        "struct_bounty_direct_obs_gold",
        "struct_global_and_plate_rule_exact_gold",
        "struct_local_est_gold",
    ]
    slice_team_gold_comps = np.zeros((N_Q, 3, 2, len(GOLD_COMP_NAMES)), dtype=np.float64)

    # 5. End-of-Window Point-in-Time States at [t0, t1, t2, t3] for each team (2: Blue, Red):
    #   0: dragon_stacks (0..4)
    #   1: has_dragon_soul (0/1)
    #   2: baron_buff_active (0/1, Baron killed within last 180s)
    #   3: elder_buff_active (0/1, Elder killed within last 150s)
    #   4: outer_turrets_standing (0..3)
    #   5: inner_turrets_standing (0..3)
    #   6: base_turrets_standing (0..3)
    #   7: nexus_turrets_standing (0..2)
    #   8: inhibitors_currently_down (0..3, destroyed within last 300s by THIS team against enemy)
    STATE_NAMES = [
        "dragon_stacks",
        "has_dragon_soul",
        "baron_buff_active",
        "elder_buff_active",
        "outer_turrets_standing",
        "inner_turrets_standing",
        "base_turrets_standing",
        "nexus_turrets_standing",
        "enemy_inhibitors_currently_down",
    ]
    window_end_states = np.zeros((N_Q, 4, 2, len(STATE_NAMES)), dtype=np.int16)

    # 6. Post-Baron Downstream Chain Tracking inside (t0, t3] (180s cumulative window):
    # For queries where a team captures Baron in (t0, t3]:
    # [baron_captured, post_baron_turrets_destroyed, post_baron_inhibs_destroyed, post_baron_kills, post_baron_buff_active_at_t3]
    post_baron_chain = np.zeros((N_Q, 2, 5), dtype=np.int16)

    q_cursor = 0
    global_m_idx = 0

    for s_idx in range(79):
        z = np.load(BANK_ROOT / f"external_{s_idx:04d}" / "BANK.npz", allow_pickle=True)
        cv = z["component_values"]
        off_cv = z["offset_component_values"]
        gt = z["gold_team"]
        off_gt = z["offset_gold_team"]
        mts = z["minute_ts"]
        off_mts = z["offset_minute_ts"]

        ev_ts = z["event_timestamp"]
        ev_tc = z["event_type_code"]
        ev_cat = z["event_category_code"]
        ev_team = z["event_acquirer_team"]
        ev_ast_cnt = z["event_assist_counts"]
        off_ev = z["offset_event_timestamp"]

        f_row = z["event_field_row"]
        f_col = z["event_field_col"]
        f_val = z["event_field_nonzero"]
        off_f = z["offset_event_field_row"]

        s_row = z["event_string_row"]
        s_col = z["event_string_col"]
        s_val = z["event_string_nonzero"]
        off_s = z["offset_event_string_row"]

        n_m = len(off_cv) - 1
        for m in range(n_m):
            c_start, c_end = int(off_cv[m]), int(off_cv[m + 1])
            if c_start == c_end:
                global_m_idx += 1
                continue

            m_cv = cv[c_start:c_end]
            prim_indices = np.where(m_cv[:, comp_cols["primary_candidate_eligible"]] == 1)[0]
            if len(prim_indices) == 0:
                global_m_idx += 1
                continue

            m_mts = mts[int(off_mts[m]):int(off_mts[m + 1])]
            m_gt = gt[int(off_gt[m]):int(off_gt[m + 1])]  # (n_frames, 2)

            e0, e1 = int(off_ev[m]), int(off_ev[m + 1])
            n_e = e1 - e0
            m_ev_ts = ev_ts[e0:e1]
            m_ev_tc = ev_tc[e0:e1]
            m_ev_cat = ev_cat[e0:e1]
            m_ev_team = ev_team[e0:e1]
            m_ev_ast = ev_ast_cnt[e0:e1]

            f0, f1 = int(off_f[m]), int(off_f[m + 1])
            r_idx, c_idx, v_arr = f_row[f0:f1], f_col[f0:f1], f_val[f0:f1]
            pid_arr = np.zeros(n_e, dtype=np.int32)
            bty_arr = np.zeros(n_e, dtype=np.int32)
            sbty_arr = np.zeros(n_e, dtype=np.int32)
            pid_arr[r_idx[c_idx == col_pid]] = v_arr[c_idx == col_pid]
            bty_arr[r_idx[c_idx == col_bty]] = v_arr[c_idx == col_bty]
            sbty_arr[r_idx[c_idx == col_sbty]] = v_arr[c_idx == col_sbty]

            s0, s1 = int(off_s[m]), int(off_s[m + 1])
            sr_idx, sc_idx, sv_arr = s_row[s0:s1], s_col[s0:s1], s_val[s0:s1]
            msub_arr = np.zeros(n_e, dtype=np.int32)
            ttype_arr = np.zeros(n_e, dtype=np.int32)
            msub_arr[sr_idx[sc_idx == scol_msub]] = sv_arr[sc_idx == scol_msub]
            ttype_arr[sr_idx[sc_idx == scol_ttype]] = sv_arr[sc_idx == scol_ttype]

            # For participant-based events (ITEM_*, WARD_*, LEVEL_UP), infer team (1..5 -> 100, 6..10 -> 200)
            ev_eff_team = m_ev_team.copy()
            pid_blue = (ev_eff_team == 0) & (pid_arr >= 1) & (pid_arr <= 5)
            pid_red = (ev_eff_team == 0) & (pid_arr >= 6) & (pid_arr <= 10)
            ev_eff_team[pid_blue] = 100
            ev_eff_team[pid_red] = 200

            # Precompute event classification masks and per-event reward components at match level (Table 1 & Table 2)
            is_kill = m_ev_tc == tc_kill
            is_drg = m_ev_cat == 1
            is_elder = is_drg & (msub_arr == elder_code)
            is_elem_drg = is_drg & (~is_elder)
            is_baron = m_ev_cat == 2
            is_herald = m_ev_cat == 3
            is_atakhan = m_ev_cat == 4
            is_horde = m_ev_cat == 5
            is_twr = m_ev_cat == 6
            is_outer = is_twr & (ttype_arr == outer_code)
            is_inner = is_twr & (ttype_arr == inner_code)
            is_base = is_twr & (ttype_arr == base_code)
            is_nexus = is_twr & (ttype_arr == nexus_code)
            is_inhib = m_ev_cat == 7
            is_plate = m_ev_cat == 8
            is_soul = m_ev_cat == 9

            is_ibuy = m_ev_tc == tc_item_buy
            is_isell = m_ev_tc == tc_item_sell
            is_iundo = m_ev_tc == tc_item_undo
            is_wplace = m_ev_tc == tc_ward_place
            is_wkill = m_ev_tc == tc_ward_kill
            is_lvlup = m_ev_tc == tc_level_up

            # First turret of match timestamp
            twr_ts_all = m_ev_ts[is_twr]
            first_twr_ts = int(twr_ts_all[0]) if len(twr_ts_all) > 0 else -1

            # Per-event gold reward components (n_e, 8)
            ev_gold = np.zeros((n_e, 8), dtype=np.float64)
            # 0: champ_kill_direct_obs_gold
            ev_gold[is_kill, 0] = (bty_arr[is_kill] + sbty_arr[is_kill]).astype(np.float64)
            # 1: champ_assist_est_gold (50% of min(bounty, 300) when assist_count > 0)
            has_ast = is_kill & (m_ev_ast > 0)
            ev_gold[has_ast, 1] = np.floor(0.5 * np.minimum(bty_arr[has_ast], 300)).astype(np.float64)
            # 2: obj_bounty_direct_obs_gold
            is_epic_any = is_drg | is_baron | is_herald | is_atakhan | is_horde
            ev_gold[is_epic_any, 2] = bty_arr[is_epic_any].astype(np.float64)
            # 3: obj_base_rule_exact_gold
            ev_gold[is_elem_drg, 3] = 25.0
            ev_gold[is_elder, 3] = 850.0
            ev_gold[is_baron, 3] = 1525.0
            ev_gold[is_herald, 3] = 100.0
            ev_gold[is_horde, 3] = 95.0
            # 4: obj_local_est_gold
            ev_gold[is_herald, 4] = 200.0
            ev_gold[is_atakhan, 4] = 100.0
            # 5: struct_bounty_direct_obs_gold
            ev_gold[is_twr | is_inhib, 5] = bty_arr[is_twr | is_inhib].astype(np.float64)
            # 6: struct_global_and_plate_rule_exact_gold
            ev_gold[is_outer, 6] = 250.0
            ev_gold[is_inner, 6] = 625.0
            ev_gold[is_base, 6] = 250.0
            ev_gold[is_nexus, 6] = 250.0
            ev_gold[is_inhib, 6] = 50.0
            ev_gold[is_plate, 6] = 125.0
            # 7: struct_local_est_gold
            ev_gold[is_outer, 7] = 250.0
            ev_gold[is_inner, 7] = 550.0
            ev_gold[is_base, 7] = 375.0
            if first_twr_ts >= 0:
                ft_mask = is_twr & (m_ev_ts == first_twr_ts)
                # Add 150G first turret bonus to first matching turret event
                ft_idx = np.where(ft_mask)[0][0]
                ev_gold[ft_idx, 7] += 150.0

            # Precompute event count matrix (n_e, N_SEV)
            ev_cnt_mat = np.zeros((n_e, N_SEV), dtype=np.int16)
            ev_cnt_mat[:, 0] = is_kill
            ev_cnt_mat[:, 1] = np.where(is_kill, m_ev_ast, 0)
            ev_cnt_mat[:, 2] = is_elem_drg
            ev_cnt_mat[:, 3] = is_elder
            ev_cnt_mat[:, 4] = is_baron
            ev_cnt_mat[:, 5] = is_atakhan
            ev_cnt_mat[:, 6] = is_herald
            ev_cnt_mat[:, 7] = is_horde
            ev_cnt_mat[:, 8] = is_outer
            ev_cnt_mat[:, 9] = is_inner
            ev_cnt_mat[:, 10] = is_base
            ev_cnt_mat[:, 11] = is_nexus
            ev_cnt_mat[:, 12] = is_inhib
            ev_cnt_mat[:, 13] = is_plate
            ev_cnt_mat[:, 14] = is_ibuy
            ev_cnt_mat[:, 15] = is_isell
            ev_cnt_mat[:, 16] = is_iundo
            ev_cnt_mat[:, 17] = is_wplace
            ev_cnt_mat[:, 18] = is_wkill
            ev_cnt_mat[:, 19] = is_lvlup

            is_team_blue = ev_eff_team == 100
            is_team_red = ev_eff_team == 200

            for local_c in prim_indices:
                row = m_cv[local_c]
                t0_idx = int(row[comp_cols["F_pre_index"]])
                t0_ms = int(row[comp_cols["F_pre_ms"]])
                L_ms = int(row[comp_cols["L"]])

                # Frame indices for [t0, t1, t2, t3]
                f_indices = [min(t0_idx + k, len(m_mts) - 1) for k in range(4)]
                f_timestamps = [int(m_mts[idx]) for idx in f_indices]
                frame_ts_4[q_cursor] = f_timestamps
                for k in range(4):
                    team_gold_4[q_cursor, k, 0] = float(m_gt[f_indices[k], 0])
                    team_gold_4[q_cursor, k, 1] = float(m_gt[f_indices[k], 1])

                # Target combat kill exchange in (t0_ms, L_ms]
                c_mask = (m_ev_ts > t0_ms) & (m_ev_ts <= L_ms) & is_kill
                if np.any(c_mask):
                    cb = c_mask & is_team_blue
                    cr = c_mask & is_team_red
                    bk = int(np.sum(cb))
                    rk = int(np.sum(cr))
                    combat_kill_exch[q_cursor, 0] = [
                        bk, rk, int(np.sum(m_ev_ast[cb])),
                        float(np.sum(ev_gold[cb, 0])), float(np.sum(ev_gold[cb, 1])),
                    ]
                    combat_kill_exch[q_cursor, 1] = [
                        rk, bk, int(np.sum(m_ev_ast[cr])),
                        float(np.sum(ev_gold[cr, 0])), float(np.sum(ev_gold[cr, 1])),
                    ]

                # 3 Marginal Slices: (t0, t1], (t1, t2], (t2, t3]
                for s_i in range(3):
                    ta, tb = f_timestamps[s_i], f_timestamps[s_i + 1]
                    if tb <= ta:
                        continue
                    smask = (m_ev_ts > ta) & (m_ev_ts <= tb)
                    if not np.any(smask):
                        continue
                    sb = smask & is_team_blue
                    sr = smask & is_team_red
                    if np.any(sb):
                        slice_team_events[q_cursor, s_i, 0] = np.sum(ev_cnt_mat[sb], axis=0)
                        slice_team_gold_comps[q_cursor, s_i, 0] = np.sum(ev_gold[sb], axis=0)
                    if np.any(sr):
                        slice_team_events[q_cursor, s_i, 1] = np.sum(ev_cnt_mat[sr], axis=0)
                        slice_team_gold_comps[q_cursor, s_i, 1] = np.sum(ev_gold[sr], axis=0)

                # End-of-window point-in-time states at t0, t1, t2, t3
                for k_f in range(4):
                    tk = f_timestamps[k_f]
                    pref = m_ev_ts <= tk
                    for t_i, t_mask in enumerate([is_team_blue, is_team_red]):
                        enemy_mask = is_team_red if t_i == 0 else is_team_blue
                        pt = pref & t_mask
                        pe = pref & enemy_mask
                        drg_st = min(4, int(np.sum(pt & is_elem_drg)))
                        has_sl = int(np.any(pt & is_soul))
                        # Baron buff active if Baron killed by team in (tk - 180000, tk]
                        bar_act = int(np.any(pt & is_baron & (m_ev_ts > tk - 180000)))
                        # Elder buff active if Elder killed by team in (tk - 150000, tk]
                        eld_act = int(np.any(pt & is_elder & (m_ev_ts > tk - 150000)))
                        # Standing turrets of THIS team = initial (3,3,3,2) minus enemy building kills <= tk
                        out_rem = max(0, 3 - int(np.sum(pe & is_outer)))
                        inn_rem = max(0, 3 - int(np.sum(pe & is_inner)))
                        bas_rem = max(0, 3 - int(np.sum(pe & is_base)))
                        nex_rem = max(0, 2 - int(np.sum(pe & is_nexus)))
                        # Enemy inhibitors currently down (destroyed by THIS team in (tk - 300000, tk])
                        inh_down = min(3, int(np.sum(pt & is_inhib & (m_ev_ts > tk - 300000))))
                        window_end_states[q_cursor, k_f, t_i] = [
                            drg_st, has_sl, bar_act, eld_act,
                            out_rem, inn_rem, bas_rem, nex_rem, inh_down,
                        ]

                # Post-Baron downstream chain inside (t0, t3]
                t3 = f_timestamps[3]
                w180_baron = (m_ev_ts > t0_ms) & (m_ev_ts <= t3) & is_baron
                if np.any(w180_baron):
                    for t_i, t_mask in enumerate([is_team_blue, is_team_red]):
                        tb_bar = w180_baron & t_mask
                        if np.any(tb_bar):
                            t_bar_ms = int(m_ev_ts[tb_bar][0])
                            after_bar = (m_ev_ts > t_bar_ms) & (m_ev_ts <= t3) & t_mask
                            post_baron_chain[q_cursor, t_i] = [
                                1,
                                int(np.sum(after_bar & is_twr)),
                                int(np.sum(after_bar & is_inhib)),
                                int(np.sum(after_bar & is_kill)),
                                int((t3 - t_bar_ms) <= 180000),
                            ]

                q_cursor += 1
            global_m_idx += 1

    assert q_cursor == N_Q, f"Expected {N_Q}, got {q_cursor}"

    # Compute Cumulative Windows (Cum 0: 0-60s, Cum 1: 0-120s, Cum 2: 0-180s) by summing Marginal Slices
    cum_team_events = np.cumsum(slice_team_events, axis=1)          # (N_Q, 3, 2, N_SEV)
    cum_team_gold_comps = np.cumsum(slice_team_gold_comps, axis=1)  # (N_Q, 3, 2, 8)

    # Observed Total Gold Increases per Team in Marginal Slices and Cumulative Windows
    slice_obs_gold_gain = np.diff(team_gold_4, axis=1)              # (N_Q, 3, 2)
    cum_obs_gold_gain = team_gold_4[:, 1:, :] - team_gold_4[:, 0:1, :]  # (N_Q, 3, 2)

    # Gold category aggregates per team:
    # - Gold_champion = comp 0 (direct_obs) + comp 1 (assist_est)
    # - Gold_objective = comp 2 (bounty_obs) + comp 3 (base_rule) + comp 4 (local_est)
    # - Gold_structure = comp 5 (bounty_obs) + comp 6 (global_and_plate_rule) + comp 7 (local_est)
    # - Gold_other_or_unallocated = Obs_Total_Gain - (Gold_champion + Gold_objective + Gold_structure)
    def gold_cat_breakdown(obs_gain: np.ndarray, comps: np.ndarray) -> np.ndarray:
        g_champ = comps[..., 0] + comps[..., 1]
        g_obj = comps[..., 2] + comps[..., 3] + comps[..., 4]
        g_struct = comps[..., 5] + comps[..., 6] + comps[..., 7]
        g_unalloc = obs_gain - (g_champ + g_obj + g_struct)
        return np.stack([obs_gain, g_champ, g_obj, g_struct, g_unalloc], axis=-1)

    slice_gold_cats = gold_cat_breakdown(slice_obs_gold_gain, slice_team_gold_comps)  # (N_Q, 3, 2, 5)
    cum_gold_cats = gold_cat_breakdown(cum_obs_gold_gain, cum_team_gold_comps)        # (N_Q, 3, 2, 5)

    # Compute Pareto Dominance Composite Verdicts for each Cumulative Window (60s, 120s, 180s) and Marginal Slice
    # Core Diff Vector (Blue - Red) across the 13 core non-kill axes:
    #   Axis 0: d_G = obs_gold_gain_Blue - obs_gold_gain_Red
    #   Axes 1..6: elem_dragon, elder_dragon, baron, atakhan, herald, horde (indices 2..7 in SLICE_EVENT_NAMES)
    #   Axes 7..12: outer_turret, inner_turret, base_turret, nexus_turret, inhibitor, plate (indices 8..13 in SLICE_EVENT_NAMES)
    def compute_verdicts_and_conflicts(obs_gain_w: np.ndarray, ev_w: np.ndarray) -> dict:
        # obs_gain_w: (N, 2), ev_w: (N, 2, N_SEV)
        d_G = obs_gain_w[:, 0] - obs_gain_w[:, 1]
        d_ev = (ev_w[:, 0, :] - ev_w[:, 1, :]).astype(np.float64)
        d_epics = d_ev[:, 2:8]      # (N, 6)
        d_structs = d_ev[:, 8:14]   # (N, 6): outer, inner, base, nexus, inhib, plate
        d_bld_no_plt = d_ev[:, 8:13] # (N, 5): buildings without plates
        d_kills = d_ev[:, 0:1]      # (N, 1)

        # V1 (Recommended Primary): Econ (1) + Epics (6) + Structures & Plates (6) = 13 axes
        mat_v1 = np.column_stack([d_G[:, None], d_epics, d_structs])
        verd_v1 = pareto_verdict_from_diffs(mat_v1)

        # V2 (Excluding Turret Plates): Econ (1) + Epics (6) + Buildings (5) = 12 axes
        mat_v2 = np.column_stack([d_G[:, None], d_epics, d_bld_no_plt])
        verd_v2 = pareto_verdict_from_diffs(mat_v2)

        # V3 (Including Kill Count as mandatory axis): V1 + Kills = 14 axes
        mat_v3 = np.column_stack([d_G[:, None], d_epics, d_structs, d_kills])
        verd_v3 = pareto_verdict_from_diffs(mat_v3)

        # Separate axis-group dominance for agreement/conflict analysis:
        verd_econ = pareto_verdict_from_diffs(d_G[:, None])
        verd_epic = pareto_verdict_from_diffs(d_epics)
        verd_struct = pareto_verdict_from_diffs(d_structs)
        verd_obj_struct = pareto_verdict_from_diffs(np.column_stack([d_epics, d_structs]))

        # Sub-classify CONFLICTED_EXCHANGE (verd_v1 == 2):
        # 1. econ_vs_obj_struct_conflict: Econ favors one team, and (Epics U Structs) has NO internal conflict and favors the other team
        # 2. within_epic_or_struct_trade: (Epics U Structs) itself is conflicted (e.g. Blue Dragon vs Red Turret/Herald)
        is_conf = verd_v1 == 2
        obj_struct_conf = is_conf & (verd_obj_struct == 2)
        econ_vs_pure_obj_struct = is_conf & (verd_obj_struct != 2)

        return {
            "verdict_v1_core13": verd_v1,
            "verdict_v2_no_plate12": verd_v2,
            "verdict_v3_with_kills14": verd_v3,
            "verdict_econ_only": verd_econ,
            "verdict_epic_only": verd_epic,
            "verdict_struct_only": verd_struct,
            "verdict_obj_struct_combined": verd_obj_struct,
            "conflict_is_within_obj_struct_trade": obj_struct_conf,
            "conflict_is_econ_vs_obj_struct": econ_vs_pure_obj_struct,
            "d_G": d_G,
        }

    cum_verdicts = [compute_verdicts_and_conflicts(cum_obs_gold_gain[:, w], cum_team_events[:, w]) for w in range(3)]
    slice_verdicts = [compute_verdicts_and_conflicts(slice_obs_gold_gain[:, s], slice_team_events[:, s]) for s in range(3)]

    # Save the complete Multi-Window Outcome Ledger NPZ (Table 3 archive)
    ledger_npz_path = LEDGER_DIR / "MULTIWINDOW_OUTCOME_LEDGER_V5.npz"
    np.savez_compressed(
        ledger_npz_path,
        match_index=q_base["match_index"],
        component_id=q_base["component_id"],
        patch=q_base["patch"],
        split_role=q_base["split_role"],
        macro_role=q_base["macro_role"],
        regime_t0=q_base["regime_t0"],
        frame_ts_4=frame_ts_4,
        team_gold_4=team_gold_4,
        combat_kill_exch=combat_kill_exch,
        slice_team_events=slice_team_events,
        cum_team_events=cum_team_events,
        slice_team_gold_comps=slice_team_gold_comps,
        cum_team_gold_comps=cum_team_gold_comps,
        slice_gold_cats=slice_gold_cats,
        cum_gold_cats=cum_gold_cats,
        window_end_states=window_end_states,
        post_baron_chain=post_baron_chain,
        cum_verdict_v1=np.column_stack([cum_verdicts[w]["verdict_v1_core13"] for w in range(3)]),
        cum_verdict_v2=np.column_stack([cum_verdicts[w]["verdict_v2_no_plate12"] for w in range(3)]),
        cum_verdict_v3=np.column_stack([cum_verdicts[w]["verdict_v3_with_kills14"] for w in range(3)]),
        slice_verdict_v1=np.column_stack([slice_verdicts[s]["verdict_v1_core13"] for s in range(3)]),
    )

    # Summarize on FIT (and compare with DEV and AUDIT)
    macro_role = q_base["macro_role"]
    regime_t0 = q_base["regime_t0"]
    end_ms = q_base["end_ms"]
    fit_mask = macro_role == "FIT"
    dev_mask = macro_role == "DEV"
    audit_mask = macro_role == "AUDIT"

    # Match winner (Blue=1, Red=0) from match_meta
    win_blue_by_match = np.zeros(len(match_meta), dtype=bool)
    # Wait: let's extract winning_team from q_base or match_table
    mt = np.load(EXT_D / "projection" / "kr_15_18_22" / "MATCH_TABLE.npy", mmap_mode="r")
    win_blue_q = mt["target_Blue_win"][q_base["match_index"]]

    def summarize_cohort(mask: np.ndarray) -> dict:
        n_q = int(np.sum(mask))
        if n_q == 0:
            return {"n_queries": 0}

        # 1. Target Combat Kill Exchange summary
        ck_b = combat_kill_exch[mask, 0, 0]
        ck_r = combat_kill_exch[mask, 1, 0]
        target_combat_summary = {
            "blue_kill_adv_rate": round(float(np.mean(ck_b > ck_r)), 4),
            "red_kill_adv_rate": round(float(np.mean(ck_r > ck_b)), 4),
            "kill_tie_rate": round(float(np.mean(ck_b == ck_r)), 4),
            "mean_blue_kills": round(float(np.mean(ck_b)), 3),
            "mean_red_kills": round(float(np.mean(ck_r)), 3),
            "mean_blue_direct_kill_gold": round(float(np.mean(combat_kill_exch[mask, 0, 3])), 2),
            "mean_red_direct_kill_gold": round(float(np.mean(combat_kill_exch[mask, 1, 3])), 2),
        }

        # 2. Marginal Slices (0-60s, 60-120s, 120-180s) & Cumulative Windows (60s, 120s, 180s)
        def summarize_window_set(is_cum: bool, idx: int, label: str) -> dict:
            obs_g = cum_obs_gold_gain[mask, idx] if is_cum else slice_obs_gold_gain[mask, idx]
            g_cats = cum_gold_cats[mask, idx] if is_cum else slice_gold_cats[mask, idx]
            g_comps = cum_team_gold_comps[mask, idx] if is_cum else slice_team_gold_comps[mask, idx]
            evs = cum_team_events[mask, idx] if is_cum else slice_team_events[mask, idx]
            vd = cum_verdicts[idx] if is_cum else slice_verdicts[idx]

            v1 = vd["verdict_v1_core13"][mask]
            v2 = vd["verdict_v2_no_plate12"][mask]
            v3 = vd["verdict_v3_with_kills14"][mask]
            vecon = vd["verdict_econ_only"][mask]
            vepic = vd["verdict_epic_only"][mask]
            vstruct = vd["verdict_struct_only"][mask]
            vobjstr = vd["verdict_obj_struct_combined"][mask]
            c_trade = vd["conflict_is_within_obj_struct_trade"][mask]
            c_econ = vd["conflict_is_econ_vs_obj_struct"][mask]
            dg = vd["d_G"][mask]

            # Absolute gold lead at end of window vs window gold change d_G
            g_end_diff = team_gold_4[mask, idx + 1, 0] - team_gold_4[mask, idx + 1, 1]
            nz_both = (g_end_diff != 0) & (dg != 0)
            abs_lead_differs_from_window_dG = float(np.mean(np.sign(g_end_diff[nz_both]) != np.sign(dg[nz_both])))

            # Agreement between Econ winner and (Epic U Struct) winner when at least one Epic/Struct diff != 0
            has_non_zero_objstr = vobjstr != 3
            objstr_unconflicted = (vobjstr == 0) | (vobjstr == 1)
            econ_agrees_with_objstr = float(np.mean(vecon[objstr_unconflicted] == vobjstr[objstr_unconflicted])) if np.any(objstr_unconflicted) else 1.0
            econ_opposes_objstr = float(np.mean(((vecon == 0) & (vobjstr == 1))[objstr_unconflicted] | ((vecon == 1) & (vobjstr == 0))[objstr_unconflicted])) if np.any(objstr_unconflicted) else 0.0

            # Match termination within window
            t_end_w = frame_ts_4[mask, idx + 1]
            ended_w = end_ms[mask] <= t_end_w
            wb_sub = win_blue_q[mask]
            n_ended = int(np.sum(ended_w))
            if n_ended > 0:
                end_wb = wb_sub[ended_w]
                end_dg = dg[ended_w]
                end_v1 = v1[ended_w]
                term_econ_agree = float(np.mean((end_wb & (end_dg > 0)) | ((~end_wb) & (end_dg < 0))))
                term_verdict_dom_agree = float(np.mean((end_wb & (end_v1 == 0)) | ((~end_wb) & (end_v1 == 1))))
                term_verdict_conflicted = float(np.mean(end_v1 == 2))
            else:
                term_econ_agree = term_verdict_dom_agree = term_verdict_conflicted = 0.0

            # Gold decomposition table (Blue mean, Red mean, Blue share of total gain, Red share of total gain)
            b_tot = float(np.mean(g_cats[:, 0, 0]))
            r_tot = float(np.mean(g_cats[:, 1, 0]))
            gold_decomp = {
                "blue_mean_gold": {
                    "observed_total_gain": round(b_tot, 2),
                    "champion_kill_assist_gold": round(float(np.mean(g_cats[:, 0, 1])), 2),
                    "epic_objective_gold": round(float(np.mean(g_cats[:, 0, 2])), 2),
                    "structure_and_plate_gold": round(float(np.mean(g_cats[:, 0, 3])), 2),
                    "other_or_unallocated_gold": round(float(np.mean(g_cats[:, 0, 4])), 2),
                    "by_source_status": {
                        "DIRECT_OBSERVED": round(float(np.mean(g_comps[:, 0, 0] + g_comps[:, 0, 2] + g_comps[:, 0, 5])), 2),
                        "RULE_EXACT": round(float(np.mean(g_comps[:, 0, 3] + g_comps[:, 0, 6])), 2),
                        "ESTIMATED_WITH_ASSUMPTION": round(float(np.mean(g_comps[:, 0, 1] + g_comps[:, 0, 4] + g_comps[:, 0, 7])), 2),
                        "OTHER_OR_UNALLOCATED": round(float(np.mean(g_cats[:, 0, 4])), 2),
                    },
                },
                "red_mean_gold": {
                    "observed_total_gain": round(r_tot, 2),
                    "champion_kill_assist_gold": round(float(np.mean(g_cats[:, 1, 1])), 2),
                    "epic_objective_gold": round(float(np.mean(g_cats[:, 1, 2])), 2),
                    "structure_and_plate_gold": round(float(np.mean(g_cats[:, 1, 3])), 2),
                    "other_or_unallocated_gold": round(float(np.mean(g_cats[:, 1, 4])), 2),
                    "by_source_status": {
                        "DIRECT_OBSERVED": round(float(np.mean(g_comps[:, 1, 0] + g_comps[:, 1, 2] + g_comps[:, 1, 5])), 2),
                        "RULE_EXACT": round(float(np.mean(g_comps[:, 1, 3] + g_comps[:, 1, 6])), 2),
                        "ESTIMATED_WITH_ASSUMPTION": round(float(np.mean(g_comps[:, 1, 1] + g_comps[:, 1, 4] + g_comps[:, 1, 7])), 2),
                        "OTHER_OR_UNALLOCATED": round(float(np.mean(g_cats[:, 1, 4])), 2),
                    },
                },
                "net_dG_blue_minus_red_mean": round(float(np.mean(dg)), 2),
                "net_dG_sd": round(float(np.std(dg)), 2),
                "blue_econ_adv_rate": round(float(np.mean(dg > 0)), 4),
                "red_econ_adv_rate": round(float(np.mean(dg < 0)), 4),
                "econ_exact_tie_rate": round(float(np.mean(dg == 0)), 6),
                "abs_total_gold_leader_differs_from_window_dG_winner_rate": round(abs_lead_differs_from_window_dG, 4),
            }

            # Per-team event rates & mean counts
            team_event_summary = {}
            for ev_i, ev_name in enumerate(SLICE_EVENT_NAMES):
                b_ev = evs[:, 0, ev_i]
                r_ev = evs[:, 1, ev_i]
                team_event_summary[ev_name] = {
                    "blue_incidence": round(float(np.mean(b_ev > 0)), 4),
                    "red_incidence": round(float(np.mean(r_ev > 0)), 4),
                    "blue_mean_count": round(float(np.mean(b_ev)), 4),
                    "red_mean_count": round(float(np.mean(r_ev)), 4),
                    "both_teams_nonzero_in_same_window_rate": round(float(np.mean((b_ev > 0) & (r_ev > 0))), 4),
                }

            # End-of-window point-in-time state summary at idx + 1
            st_k = window_end_states[mask, idx + 1]  # (n_q, 2, 9)
            state_summary = {}
            for st_i, st_name in enumerate(STATE_NAMES):
                state_summary[st_name] = {
                    "blue_mean": round(float(np.mean(st_k[:, 0, st_i])), 4),
                    "red_mean": round(float(np.mean(st_k[:, 1, st_i])), 4),
                }

            return {
                "window_label": label,
                "economic_outcome": gold_decomp,
                "pareto_composite_verdict_v1_core13": {
                    "BLUE_DOMINANT": round(float(np.mean(v1 == 0)), 4),
                    "RED_DOMINANT": round(float(np.mean(v1 == 1)), 4),
                    "CONFLICTED_EXCHANGE": round(float(np.mean(v1 == 2)), 4),
                    "NO_DIFFERENCE": round(float(np.mean(v1 == 3)), 6),
                    "conflict_breakdown": {
                        "econ_vs_unconflicted_obj_struct_rate": round(float(np.mean(c_econ)), 4),
                        "within_obj_or_struct_asymmetric_trade_rate": round(float(np.mean(c_trade)), 4),
                    },
                },
                "pareto_composite_verdict_v2_no_plates12": {
                    "BLUE_DOMINANT": round(float(np.mean(v2 == 0)), 4),
                    "RED_DOMINANT": round(float(np.mean(v2 == 1)), 4),
                    "CONFLICTED_EXCHANGE": round(float(np.mean(v2 == 2)), 4),
                    "NO_DIFFERENCE": round(float(np.mean(v2 == 3)), 6),
                },
                "pareto_composite_verdict_v3_with_kills14": {
                    "BLUE_DOMINANT": round(float(np.mean(v3 == 0)), 4),
                    "RED_DOMINANT": round(float(np.mean(v3 == 1)), 4),
                    "CONFLICTED_EXCHANGE": round(float(np.mean(v3 == 2)), 4),
                    "NO_DIFFERENCE": round(float(np.mean(v3 == 3)), 6),
                },
                "axis_agreement_diagnostics": {
                    "queries_with_any_obj_or_struct_diff_rate": round(float(np.mean(has_non_zero_objstr)), 4),
                    "queries_with_unconflicted_obj_struct_diff_rate": round(float(np.mean(objstr_unconflicted)), 4),
                    "econ_agrees_with_unconflicted_obj_struct_rate": round(econ_agrees_with_objstr, 4),
                    "econ_opposes_unconflicted_obj_struct_rate": round(econ_opposes_objstr, 4),
                },
                "match_termination_outcome": {
                    "ended_within_window_rate": round(float(np.mean(ended_w)), 4),
                    "ended_queries_n": n_ended,
                    "among_ended_game_winner_matches_econ_sign_rate": round(term_econ_agree, 4),
                    "among_ended_game_winner_matches_pareto_dominant_rate": round(term_verdict_dom_agree, 4),
                    "among_ended_pareto_conflicted_rate": round(term_verdict_conflicted, 4),
                },
                "team_events_blue_and_red": team_event_summary,
                "end_of_window_point_in_time_states": state_summary,
            }

        marginal_slices = {
            "SLICE_1_0_to_60s": summarize_window_set(False, 0, "Marginal 0~60s (t0, t1]"),
            "SLICE_2_60_to_120s": summarize_window_set(False, 1, "Marginal 60~120s (t1, t2]"),
            "SLICE_3_120_to_180s": summarize_window_set(False, 2, "Marginal 120~180s (t2, t3]"),
        }
        cumulative_windows = {
            "CUM_60s_0_to_60s": summarize_window_set(True, 0, "Cumulative 0~60s (t0, t1]"),
            "CUM_120s_0_to_120s": summarize_window_set(True, 1, "Cumulative 0~120s (t0, t2]"),
            "CUM_180s_0_to_180s": summarize_window_set(True, 2, "Cumulative 0~180s (t0, t3]"),
        }

        # 3. Window-to-Window Transition Matrices (60s -> 120s -> 180s) for Cumulative Verdict V1 and Economic Sign
        v60 = cum_verdicts[0]["verdict_v1_core13"][mask]
        v120 = cum_verdicts[1]["verdict_v1_core13"][mask]
        v180 = cum_verdicts[2]["verdict_v1_core13"][mask]

        def trans_mat(va: np.ndarray, vb: np.ndarray) -> dict:
            out = {}
            for i, iname in VERDICT_NAMES.items():
                mi = va == i
                ni = int(np.sum(mi))
                if ni == 0:
                    continue
                out[iname] = {
                    "n_from": ni,
                    "share_from": round(ni / len(va), 4),
                    "to_probs": {
                        jname: round(float(np.mean(vb[mi] == j)), 4)
                        for j, jname in VERDICT_NAMES.items()
                    },
                }
            return out

        traj_counts = collections.Counter(zip(v60.tolist(), v120.tolist(), v180.tolist()))
        top_trajectories = []
        for (a, b, c), cnt in traj_counts.most_common(15):
            top_trajectories.append({
                "trajectory": f"60s:{VERDICT_NAMES[a]} -> 120s:{VERDICT_NAMES[b]} -> 180s:{VERDICT_NAMES[c]}",
                "count": int(cnt),
                "share": round(cnt / n_q, 4),
            })

        # Post-Baron downstream chain summary
        pb = post_baron_chain[mask]  # (n_q, 2, 5)
        bar_any_b = pb[:, 0, 0] == 1
        bar_any_r = pb[:, 1, 0] == 1
        n_bar_captures = int(np.sum(bar_any_b) + np.sum(bar_any_r))
        if n_bar_captures > 0:
            tw_after = np.concatenate([pb[bar_any_b, 0, 1], pb[bar_any_r, 1, 1]])
            in_after = np.concatenate([pb[bar_any_b, 0, 2], pb[bar_any_r, 1, 2]])
            kl_after = np.concatenate([pb[bar_any_b, 0, 3], pb[bar_any_r, 1, 3]])
            bf_rem = np.concatenate([pb[bar_any_b, 0, 4], pb[bar_any_r, 1, 4]])
            baron_chain_summary = {
                "n_team_baron_captures_in_180s": n_bar_captures,
                "post_baron_turret_destroyed_incidence": round(float(np.mean(tw_after > 0)), 4),
                "post_baron_turret_mean_count": round(float(np.mean(tw_after)), 3),
                "post_baron_inhibitor_destroyed_incidence": round(float(np.mean(in_after > 0)), 4),
                "post_baron_kills_mean_count": round(float(np.mean(kl_after)), 3),
                "baron_buff_still_active_at_t3_rate": round(float(np.mean(bf_rem > 0)), 4),
            }
        else:
            baron_chain_summary = {"n_team_baron_captures_in_180s": 0}

        return {
            "n_queries": n_q,
            "target_combat_kill_exchange": target_combat_summary,
            "marginal_slices": marginal_slices,
            "cumulative_windows": cumulative_windows,
            "verdict_transitions": {
                "same_verdict_60s_to_120s_rate": round(float(np.mean(v60 == v120)), 4),
                "same_verdict_120s_to_180s_rate": round(float(np.mean(v120 == v180)), 4),
                "same_verdict_all_three_60_120_180_rate": round(float(np.mean((v60 == v120) & (v120 == v180))), 4),
                "dominant_flip_blue_to_red_or_red_to_blue_60_to_120_rate": round(float(np.mean(((v60 == 0) & (v120 == 1)) | ((v60 == 1) & (v120 == 0)))), 4),
                "dominant_to_conflicted_60_to_120_rate": round(float(np.mean(((v60 == 0) | (v60 == 1)) & (v120 == 2))), 4),
                "conflicted_to_dominant_60_to_120_rate": round(float(np.mean((v60 == 2) & ((v120 == 0) | (v120 == 1)))), 4),
                "transition_matrix_60s_to_120s": trans_mat(v60, v120),
                "transition_matrix_120s_to_180s": trans_mat(v120, v180),
                "top_15_full_trajectories_60_120_180": top_trajectories,
            },
            "post_baron_downstream_chain_in_180s": baron_chain_summary,
        }

    fit_summary = summarize_cohort(fit_mask)
    dev_summary = summarize_cohort(dev_mask)
    audit_summary = summarize_cohort(audit_mask)

    fit_by_regime = {}
    for r_id, r_name in REGIME_NAMES.items():
        fit_by_regime[r_name] = summarize_cohort(fit_mask & (regime_t0 == r_id))

    # 6. Multi-Target Input History (L in {1, 2, 3, 4, 5} frames) Baseline Comparison on FIT -> DEV
    # Compare predicting the exact same multi-target vector Y_H at H=120s across L=1..5:
    # - Regression targets (MAE):
    #   (a) Total net gold change d_G(120s)
    #   (b) Event-category gold diff: d_Gold_champion(120s), d_Gold_objective(120s), d_Gold_structure(120s)
    #   (c) Structure count diff: d_Turrets(120s)
    # - Probability / Classification targets (Brier score):
    #   (d) Any Epic Objective capture by Blue vs Red in 120s
    #   (e) Any Turret destruction by Blue vs Red in 120s
    #   (f) End-of-window state: Blue Baron/Dragon Soul advantage at t2
    #   (g) 4-Class Pareto Composite Verdict (BLUE_DOMINANT, RED_DOMINANT, CONFLICTED_EXCHANGE) one-vs-rest Brier
    fit_hms = q_base["hist_frame_ms"][fit_mask]
    fit_hgd = q_base["hist_g_diff"][fit_mask]
    fit_hkd = q_base["hist_kills_diff"][fit_mask]
    fit_avail = q_base["avail_t0"][fit_mask]
    fit_t0 = q_base["t0_ms"][fit_mask]
    fit_st0 = (window_end_states[fit_mask, 0, 0, :] - window_end_states[fit_mask, 0, 1, :]).astype(np.float64)

    dev_hms = q_base["hist_frame_ms"][dev_mask]
    dev_hgd = q_base["hist_g_diff"][dev_mask]
    dev_hkd = q_base["hist_kills_diff"][dev_mask]
    dev_avail = q_base["avail_t0"][dev_mask]
    dev_t0 = q_base["t0_ms"][dev_mask]
    dev_st0 = (window_end_states[dev_mask, 0, 0, :] - window_end_states[dev_mask, 0, 1, :]).astype(np.float64)
    dev_midx = q_base["match_index"][dev_mask]

    unique_dev_m, dev_inv = np.unique(dev_midx, return_inverse=True)
    n_dev_m = len(unique_dev_m)
    dev_q_per_m = np.bincount(dev_inv, minlength=n_dev_m).astype(np.float64)
    dev_w_q = 1.0 / np.maximum(1.0, dev_q_per_m[dev_inv])

    def build_X(hms, hgd, hkd, avail, t0_s, st0, L_f: int) -> np.ndarray:
        cols = [
            np.ones(len(hms), dtype=np.float64),
            hgd[:, 0] / 1000.0,
            hkd[:, 0] / 10.0,
            t0_s / 600000.0,
            avail.astype(np.float64),
            st0 / 3.0,
        ]
        for k in range(1, L_f):
            mk = (hms[:, k] >= 0).astype(np.float64)
            dg_k = ((hgd[:, 0] - hgd[:, k]) / 1000.0) * mk
            dk_k = ((hkd[:, 0] - hkd[:, k]) / 10.0) * mk
            cols.extend([mk, dg_k, dk_k])
        return np.column_stack(cols)

    # Targets at H=120s (w=1) and H=180s (w=2)
    multi_target_L_comparison = {}
    for w_idx, w_name in [(0, "CUM_60s"), (1, "CUM_120s"), (2, "CUM_180s")]:
        # Extract targets for FIT and DEV
        f_gc = cum_gold_cats[fit_mask, w_idx, 0] - cum_gold_cats[fit_mask, w_idx, 1]  # (N_fit, 5)
        d_gc = cum_gold_cats[dev_mask, w_idx, 0] - cum_gold_cats[dev_mask, w_idx, 1]  # (N_dev, 5)

        f_ev_b = cum_team_events[fit_mask, w_idx, 0]
        f_ev_r = cum_team_events[fit_mask, w_idx, 1]
        d_ev_b = cum_team_events[dev_mask, w_idx, 0]
        d_ev_r = cum_team_events[dev_mask, w_idx, 1]

        # Binary indicator targets: Blue captures any epic, Blue destroys any turret, Pareto Verdict classes (0, 1, 2)
        f_b_epic = (np.sum(f_ev_b[:, 2:8], axis=1) > 0).astype(np.float64)
        d_b_epic = (np.sum(d_ev_b[:, 2:8], axis=1) > 0).astype(np.float64)
        f_b_twr = (np.sum(f_ev_b[:, 8:12], axis=1) > 0).astype(np.float64)
        d_b_twr = (np.sum(d_ev_b[:, 8:12], axis=1) > 0).astype(np.float64)
        f_twr_cnt_diff = (np.sum(f_ev_b[:, 8:12], axis=1) - np.sum(f_ev_r[:, 8:12], axis=1)).astype(np.float64)
        d_twr_cnt_diff = (np.sum(d_ev_b[:, 8:12], axis=1) - np.sum(d_ev_r[:, 8:12], axis=1)).astype(np.float64)

        f_v1 = cum_verdicts[w_idx]["verdict_v1_core13"][fit_mask]
        d_v1 = cum_verdicts[w_idx]["verdict_v1_core13"][dev_mask]
        f_v_onehot = np.column_stack([(f_v1 == c).astype(np.float64) for c in range(3)])
        d_v_onehot = np.column_stack([(d_v1 == c).astype(np.float64) for c in range(3)])

        Y_fit_all = np.column_stack([f_gc[:, 0:4], f_twr_cnt_diff, f_b_epic, f_b_twr, f_v_onehot])
        Y_dev_all = np.column_stack([d_gc[:, 0:4], d_twr_cnt_diff, d_b_epic, d_b_twr, d_v_onehot])

        by_L = {}
        for L_f in range(1, 6):
            X_tr = build_X(fit_hms, fit_hgd, fit_hkd, fit_avail, fit_t0, fit_st0, L_f)
            X_dv = build_X(dev_hms, dev_hgd, dev_hkd, dev_avail, dev_t0, dev_st0, L_f)
            reg = np.eye(X_tr.shape[1]) * 10.0
            reg[0, 0] = 0.0
            B = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ Y_fit_all)
            P = X_dv @ B

            # Match-weighted MAE for continuous targets (cols 0..4) and Brier for probability targets (cols 5..9)
            def mw_mae(col_i: int) -> float:
                err = np.abs(Y_dev_all[:, col_i] - P[:, col_i])
                return round(float(np.mean(np.bincount(dev_inv, weights=err * dev_w_q, minlength=n_dev_m))), 3)

            def mw_brier(col_i: int) -> float:
                p_clip = np.clip(P[:, col_i], 0.001, 0.999)
                sq = (Y_dev_all[:, col_i] - p_clip) ** 2
                return round(float(np.mean(np.bincount(dev_inv, weights=sq * dev_w_q, minlength=n_dev_m))), 5)

            p3 = np.clip(P[:, 7:10], 0.001, 0.999)
            p3 = p3 / np.sum(p3, axis=1, keepdims=True)
            v_brier = np.sum((d_v_onehot - p3) ** 2, axis=1)
            mw_v_brier = round(float(np.mean(np.bincount(dev_inv, weights=v_brier * dev_w_q, minlength=n_dev_m))), 5)

            by_L[f"L{L_f}_frames"] = {
                "total_gold_dG_mae": mw_mae(0),
                "champion_gold_dG_mae": mw_mae(1),
                "objective_gold_dG_mae": mw_mae(2),
                "structure_gold_dG_mae": mw_mae(3),
                "turret_count_diff_mae": mw_mae(4),
                "blue_epic_capture_brier": mw_brier(5),
                "blue_turret_destroy_brier": mw_brier(6),
                "pareto_verdict_3class_brier": mw_v_brier,
            }
        multi_target_L_comparison[w_name] = by_L

    out_json = {
        "version": "multiwindow-outcome-ledger-v5-r1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_executed": False,
        "ledger_archive": {
            "path": str(ledger_npz_path.relative_to(PREP_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(ledger_npz_path),
            "rows": N_Q,
        },
        "reward_provenance_taxonomy": {
            "DIRECT_OBSERVED": [
                "CHAMPION_KILL: bounty + shutdownBounty (100% present in Match-V5, paid to killerId)",
                "ELITE_MONSTER_KILL: bounty (non-zero in 1.0-2.5% of kills when Objective Bounty active)",
                "BUILDING_KILL: bounty (non-zero in 3.21% of turret kills when Objective Bounty active)",
                "ITEM_UNDO: goldGain (shop refund/deduction)",
            ],
            "RULE_EXACT": [
                "ELITE_MONSTER_KILL base team/killer gold: Elemental Dragon 25G, Elder Dragon 850G (100+150x5), Baron Nashor 1525G (25+300x5), Rift Herald killer 100G, Voidgrub 95G (20+15x5)",
                "BUILDING_KILL global team gold: Outer Turret 250G (50x5), Inner Turret 625G (125x5), Base Turret 250G (50x5), Nexus Turret 250G (50x5), Inhibitor 50G",
                "TURRET_PLATE_DESTROYED: 125G per plate to acquiring team",
            ],
            "ESTIMATED_WITH_ASSUMPTION": [
                "CHAMPION_KILL assist pool: floor(0.5 * min(bounty, 300)) when assist_count >= 1 (standard S15 rule; excludes Pyke R / Draven passive / Treasure Hunter)",
                "ELITE_MONSTER_KILL local/approx gold: Rift Herald local 200G, Atakhan ~100G",
                "BUILDING_KILL local proximity gold: Outer 250G, Inner 550G, Base 375G + First Turret 150G (assumes >=1 champion in local proximity at destruction)",
            ],
            "UNVERIFIED_MISSING_AND_RESIDUAL": [
                "Individual local gold split among nearby champions between minute frames (kept as team-level only, not guessed per player)",
                "Champion/rune bonus gold (Draven passive, Pyke R Your Cut, First Strike, Cash Back, Treasure Hunter, Cull)",
                "other_or_unallocated_gold = Observed_Team_Gold_Gain - (Gold_champion + Gold_objective + Gold_structure): includes minion/jungle CS, passive generation, ward kills, item sales, and frame-timing/proximity residuals",
            ],
        },
        "fit_overall": fit_summary,
        "fit_by_t0_objective_regime": fit_by_regime,
        "out_of_time_stability": {
            "DEV_15_21": dev_summary,
            "AUDIT_15_21": audit_summary,
        },
        "multi_target_input_window_L1_to_L5_baseline_FIT_to_DEV": multi_target_L_comparison,
    }

    write_json(LEDGER_DIR / "MULTIWINDOW_OUTCOME_LEDGER_SUMMARY_V5.json", out_json)
    print("Successfully built multiwindow_outcome_ledger!")


if __name__ == "__main__":
    main()
