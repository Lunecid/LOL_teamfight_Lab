"""Generate REPORT.md and DEFINITION_AND_EVIDENCE.md for outputs/cohort_role_training_20260915 from saved artifacts only."""
from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

O = K.OUT


def J(p):
    p = O / p
    return C.read_json(p) if p.exists() else None


def f(x, d=4):
    return 'NA' if x is None else f'{x:.{d}f}'


def n(x):
    return 'NA' if x is None else f'{x:,}'


def diff(b, k, d=5):
    if not b:
        return 'NA'
    x = b['a_minus_b'][k]
    if x['ci95'] is None:
        return f'{x["estimate"]:+.{d}f}'
    return f'{x["estimate"]:+.{d}f} [{x["ci95"][0]:+.{d}f}, {x["ci95"][1]:+.{d}f}]'


def pair(cc, a, b):
    for p in (cc.get('bootstrap') or {}).get('pairs', []):
        if p['a'] == a and p['b'] == b:
            return p
    return None


SETS = ['MAIN_TEST', 'EXT_KR_16.13', 'EXT_KR_16.14_pilot', 'EXT_KR_16.15', 'EXT_NA1_16.13']


def sig(p, k):
    """'better' / 'worse' / 'n.s.' for a minus b by the 95% CI (Brier/log loss: negative better; AUC: positive better)."""
    if not p or not p['a_minus_b'][k]['ci95']:
        return 'NA'
    lo, hi = p['a_minus_b'][k]['ci95']
    good = (hi < 0) if k != 'auc' else (lo > 0)
    bad = (lo > 0) if k != 'auc' else (hi < 0)
    return 'better' if good else ('worse' if bad else 'n.s.')


def key_results(w, RA, RC, rel, sh, rman):
    w('## Key results (h90 primary; all TEST numbers carry prior exposure)')
    w('')
    t = RA['MAIN_TEST']['h90']
    sT = C.read_json(O / 'selection' / 'q_specialist_T_h90.json')
    sN = C.read_json(O / 'selection' / 'q_specialist_N_h90.json')

    def pool_sel(s):
        return s['pooled_reference']['DIAGNOSTIC_cohort_validation_metrics_not_used_for_selection']['select']['brier']
    bT = pair(t['cohorts']['T'], 'specialist', 'pooled')
    bN = pair(t['cohorts']['N'], 'specialist', 'pooled')
    w(f'1. **Teamfight specialist (A).** On the 32,981 identical TEST T rows the T specialist ({sT["chosen"]}) beats the pooled q: Brier '
      f'{diff(bT, "brier")}, log loss {diff(bT, "logloss")}, AUC {diff(bT, "auc", 4)}; its Q_SELECT Brier was also lower '
      f'({sT["select_metrics"][sT["chosen"]]["brier"]:.6f} vs pooled {pool_sel(sT):.6f}). The same direction holds at h60/h120 on TEST. On external sets '
      'at h90 the T differences are not significant (KR 16.13 point estimate favours the specialist; NA1 16.13 Brier point estimate favours pooled), '
      'and at h60 the NA1 16.13 T specialist is significantly worse in Brier and log loss, so the T gain does not transfer reliably to 16.xx.')
    w(f'2. **Non-teamfight specialist (A).** On TEST N rows: Brier {diff(bN, "brier")}, log loss {diff(bN, "logloss")} (n.s.), AUC {diff(bN, "auc", 4)}. '
      f'This gain is small and not supported on validation: on N Q_SELECT the pooled q was better ({pool_sel(sN):.6f} vs specialist '
      f'{sN["select_metrics"][sN["chosen"]]["brier"]:.6f}; diagnostic, not used for selection). The h60 N specialist ({C.read_json(O / "selection" / "q_specialist_N_h60.json")["chosen"]}) '
      'is worse in AUC on TEST. Treat N specialization as not robust.')
    o = t['oracle_routing_NOT_DEPLOYABLE']
    w(f'3. **Oracle-cohort routing** (post-cutoff membership, not deployable): TEST Brier {diff(pair(o, "oracle_routed", "pooled"), "brier")} vs pooled.')
    cT, cN = RC['MAIN_TEST']['T'], RC['MAIN_TEST']['N']
    w(f'4. **Role representation (C), N cohort.** With the same LightGBM base and calibration family, TEST N: role - participant Brier '
      f'{diff(pair(cN, "role_winner", "participant_winner"), "brier")}, draft control - participant {diff(pair(cN, "draft_winner", "participant_winner"), "brier")}, '
      f'role - draft control {diff(pair(cN, "role_winner", "draft_winner"), "brier")}. On external N rows role - participant Brier is '
      f'{diff(pair(RC["EXT_KR_16.13"]["N"], "role_winner", "participant_winner"), "brier")} in KR 16.13 '
      f'({sig(pair(RC["EXT_KR_16.13"]["N"], "role_winner", "participant_winner"), "brier")}) and '
      f'{diff(pair(RC["EXT_NA1_16.13"]["N"], "role_winner", "participant_winner"), "brier")} in NA1 16.13 '
      f'({sig(pair(RC["EXT_NA1_16.13"]["N"], "role_winner", "participant_winner"), "brier")}); the two small sets are n.s. So role organization of the same '
      'participant information adds signal beyond slot order and beyond direct draft indicators for non-teamfight engagements, as ranked on Q_SELECT before TEST.')
    w(f'5. **Role representation (C), T cohort.** No TEST arm pair is significant (role - participant {diff(pair(cT, "role_winner", "participant_winner"), "brier")}); '
      f'the overall T winner ({cT["overall"]}) does not beat the A specialist ridge on TEST ({diff(pair(cT, "overall_winner", "A_specialist"), "brier")}).')
    w(f'6. **Model class.** The new full-feature LightGBM is the within-arm winner everywhere; in N the overall winner beats the A specialist ridge by '
      f'{diff(pair(cN, "overall_winner", "A_specialist"), "brier")} Brier on TEST, a mix of model class and representation effects.')
    abT = [(g_, pair(cT, f'drop_{g_}', 'full_role_ridge_preselected')) for g_ in ('role_TOP', 'role_JUNGLE', 'role_MIDDLE', 'role_BOTTOM', 'role_UTILITY')]
    w('7. **Role-block ablations (T, role ridge).** TEST Brier degradation when a block is removed: ' + '; '.join(f'{g_[5:]} {diff(p, "brier")}' for g_, p in abT)
      + '. Removing TOP or UTILITY (and, negligibly, BOTTOM) lowered Q_SELECT Brier, and external results are inconsistent, so blocks are partly redundant with each other and with global totals; this is a model ablation, not causal lane importance.')
    accs = ', '.join(f'{s} {v["posterior_argmax_accuracy"]:.3f}' for s, v in rel['results'].items())
    ag = rman['outputs']['MAIN_TRAIN']['OOF_agreement_with_WEAK_proxy_NOT_validated_accuracy']['posterior_argmax_agreement']
    w(f'8. **Role estimates.** Draft-only role posteriors agree with the weak TRAIN proxy at {ag:.3f} (OOF; not accuracy). On independent external raw teamPosition '
      f'(post-freeze) participant accuracy is {accs}; unseen champions are harder (section 5).')
    shT, shN = sh['cohorts']['T'], sh['cohorts']['N']
    w('9. **Explanations (descriptive).** Mean |group Shapley| on the final probability: T ' + ', '.join(f'{k} {v:.4f}' for k, v in shT['global_mean_abs'].items())
      + '; N ' + ', '.join(f'{k} {v:.4f}' for k, v in shN['global_mean_abs'].items()) + '. Global context dominates T; in N (more than half of the explained '
      'rows start before 10 min) the BOTTOM and JUNGLE role groups are comparable to global context (256-row descriptive sample per cohort).')
    w('')


def main():
    cm, neg, fx = J('cohorts/cohort_manifest.json'), J('cohorts/negative_count_provenance.json'), J('cohorts/detector_train_fixture.json')
    prov, dman, proto, fz = J('role_supervision_provenance.json'), J('draft/draft_manifest.json'), J('protocol.json'), J('frozen_manifest.json')
    rman, ct, val = J('role_models_manifest.json'), J('contract_tests/result.json'), J('validation.json')
    RA = J('eval/results_A.json')['results']
    RCj = J('eval/results_C.json')
    RC = RCj['results'] if RCj else {}
    rel = J('eval/role_reliability_external_raw.json')
    sh = J('shap/shap_summary.json')
    diag = J('diagnostics/ridge_optimizer_path_diagnostic.json')
    snap = J('integrity/snapshot_diff.json')
    L = []
    w = L.append
    w('# Teamfight / non-teamfight q specialists and role-aware q representation (2026-09-15)')
    w('')
    w(f'Generated {time.strftime("%Y-%m-%d %H:%M:%S")} KST from saved artifacts. Design: docs/CLAUDE_COHORT_ROLE_TRAIN_20260915.md (Codex). '
      'Implementation and execution: Claude Opus 5. Output root: outputs/cohort_role_training_20260915. Read-only source: '
      'outputs/full_corpus_training_20260915 (pre states, keys, labels, V, pooled q).')
    w('')
    w('**Role of this work.** Exploratory follow-up on model-defined labels (Y = 1[V(endpoint) - V(q_pre) > 0], estimated '
      'probability improvement, not ground truth). TEST 15.16 and every external set were already evaluated for the pooled q in the '
      'completed full run (and earlier work), so all TEST numbers here are **frozen re-evaluations with prior exposure, not a fresh '
      'confirmatory test**. Every new choice (cohort specialists, role models, arms, ablations) was frozen before this task opened '
      'any TEST/external label. Cohort membership uses post-cutoff participation and is not a live input.')
    w('')
    key_results(w, RA, RC, rel, sh, rman)
    # ------------------------------------------------------------------ 1 status
    w('## 1. Stage status')
    w('')
    w('| Stage | State | Evidence |')
    w('|---|---|---|')
    sd = val['checks'] if val else {}
    w(f'| Read-only integrity snapshot before / after | executed; verified ({"unchanged" if val and all(v for k, v in sd.items() if k.startswith("readonly_")) else "see validation"}) | integrity/snapshot_before.json, snapshot_after.json, snapshot_diff.json |')
    w(f'| A1 scale restoration + cohort manifest (before fitting) | executed ({cm["written_at"]}); verified | cohorts/cohort_manifest.json |')
    w(f'| B0 draft fields + weak role-supervision provenance (before fitting) | executed ({prov["written_at"]}) | draft/draft_manifest.json, role_supervision_provenance.json |')
    w(f'| Contract tests (TRAIN-only / synthetic) | run 1 failed (5 fixture/code defects fixed), run 2 and run 3 passed ({ct["passed_count"]} tests, gate run {ct["run"]} at {ct["at"]}) | contract_tests/run1-3.txt, result.json |')
    w(f'| Protocol (predeclared rules and pairs) | executed ({proto["written_at"]}) | protocol.json |')
    w('| TRAIN-only smoke runs (1/8 TRAIN match subset, pseudo roles; code paths only) | executed; run 1 of arms failed (ridge reload identity), fixed | smoke_train_only/, logs/smoke_* |')
    w(f'| A2 T/N specialist q (h90/h60/h120) | executed | selection/q_specialist_*.json |')
    w(f'| B1 role models (final + 5 OOF) | executed; converged | role_models_manifest.json |')
    w(f'| C arms (T, N; h90) + T ablations | executed | selection/arms_*_h90.json, ablations_T_h90.json |')
    w(f'| Freeze | executed ({fz["frozen_at"]}); {len(fz["frozen_files_sha256"])} files hashed | frozen_manifest.json |')
    w('| Frozen evaluation (TEST + 4 external sets) | executed; frozen hashes unchanged | eval/results_A.json, results_C.json, hashes_*.json |')
    w('| Role reliability vs external raw teamPosition (post-freeze) | executed | eval/role_reliability_external_raw.json |')
    w(f'| Role-arm group SHAP | executed; checks {"pass" if sh and sh["all_checks_pass"] else "FAIL"} | shap/shap_summary.json |')
    w('| Post-freeze ridge optimizer diagnostic (TRAIN/VALIDATION only) | executed | diagnostics/ridge_optimizer_path_diagnostic.json |')
    w(f'| Post-run verification | {"pass" if val and not val["failed"] else "FAILED: " + ", ".join(val["failed"]) if val else "not run"} ({len(sd)} checks) | validation.json |')
    w('')
    # ------------------------------------------------------------------ 2 scale
    w('## 2. Scale definition restored: v3.3 participation, teamfight cut 4')
    w('')
    w('Authoritative scale block (D:/LOL_Project/fusion_2615/corpus_shards_v33/manifest.json): '
      f'`{cm["inputs"]["shard_manifest_scale"]}`; detector `{cm["inputs"]["shard_manifest_detector"]}` (G 13.7 s, D 4,264 u, R 1,600 u, B 15 s; M 2). '
      'The old ENGAGEMENT_SCALE_DEFINITION.md (v2, cut 3) was not used.')
    w('')
    w('* **Participation count** = stored `cluster_blue` / `cluster_red` (gameplay/fights.py `det_cluster_*`): champions among the killers, '
      'victims and assisters of the engagement\'s kills plus actors of other timeline events within 3,000 u of the anchor between the start and '
      'the last kill (+ tail), structure/monster events excluded, shop events included (`TF2_EXCLUDE_SHOP_INTERACTIONS` False), position-less '
      'events at the actor\'s interpolated position; a merged engagement keeps its earlier candidate\'s counts. It is **not** the pre-cutoff presence '
      'count (`present_*`, diagnostic only).')
    w(f'* **Recovery:** all 32 shards joined by exact (match, engage_ts = s) key to every frozen label row: {n(cm["join"]["shard_rows_used_once"])} of '
      f'{n(cm["join"]["shard_rows"])} shard rows used exactly once, {n(cm["join"]["label_rows"])} label rows, valid rows h60/h90/h120 = '
      f'{n(cm["join"]["valid_rows"]["h90"])}; label (s, L) equal the parent exposure (s, L) for every match; no shard omission, so no main-corpus '
      're-detection was needed for counts. Old market_event labels, y and X were not read.')
    w(f'* **Negative counts:** {neg["rows_total"]} main rows carry -1. `_fight_to_ref_row` stores `int(count or -1)`. The frozen detector was re-run on '
      f'those {neg["matches"]} matches: every row re-detected once at the same (s, L) with raw integer count 0 on the -1 side '
      f'({neg["proven_rows"]} proven, {neg["unknown_rows"]} unknown), so -1 is read as 0. External sets: '
      + ', '.join(f'{k} {v["rows_negative_count"]} (proven {v["rows_negative_count_proven_zero"]})' for k, v in cm['external_checks'].items())
      + '. No unknown-scale row remains in any set; the unknown category is implemented but empty.')
    w(f'* **Detector checks (TRAIN-only fixture):** {fx["n"]} hash-selected 15.14 matches re-detected: exposure tuples equal for '
      f'{fx["counts"]["exposure_tuple_equal"]}/{fx["n"]} matches; participation and presence counts equal the shards for '
      f'{fx["counts"]["counts_equal_shard"]}/{fx["counts"]["engagements"]} engagements; settings {fx["detector_settings"]}.')
    w('* **External sets:** the same detector re-run on the already-adapted caches regenerated the participation fields; exposure tuples equal the '
      'stored external exposures bitwise and label keys equal the re-detected keys for all four sets ('
      + ', '.join(f'{k}: {v["matches"]} matches, {n(v["label_rows"])} rows' for k, v in cm['external_checks'].items()) + ').')
    w('')
    w('E = all engagement rows; T = {min(blue, red) >= 4}; N = E \\ T (known scale). T and N are disjoint and their union is the known-scale E in every set.')
    w('')
    w('| Set / role | rows | valid h90 | T (valid) | N (valid) | pick <=1 (valid) | skirmish 2-3 (valid) | T share | cut 3 teamfight (DIAG, all rows) | cut 5 (DIAG) | presence min >= 4 (DIAG) |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for nm, s in cm['sets'].items():
        for sr, c in s['by_sub_role'].items():
            h = c['h90']
            w(f'| {nm} {sr} | {n(c["all_rows"]["rows"])} | {n(h["valid"])} | {n(h["T"])} | {n(h["N"])} | {n(h["pick"])} | {n(h["skirmish"])} | {h["T"] / max(1, h["valid"]):.3f} | | | |')
        w(f'| **{nm} total** | {n(s["rows"])} | | {n(s["T"])} (all rows) | {n(s["N"])} | {n(s["pick"])} | {n(s["skirmish"])} | {s["T"] / s["rows"]:.3f} | {n(s["cut3_teamfight_DIAG"])} | {n(s["cut5_teamfight_DIAG"])} | {n(s["presence_n_min_ge4_DIAG"])} |')
    tot_T = sum(cm['sets'][k]['T'] for k in ('MAIN_TRAIN', 'MAIN_VALIDATION', 'MAIN_TEST'))
    w('')
    w(f'Main corpus at cut 4: {n(tot_T)} teamfight rows of 566,452 detected engagements. The manuscript\'s 109,829 teamfights refer to the '
      'market_event-labelled population (532,547 rows, draws dropped); the deltaV population here is every detected engagement, so the counts '
      'differ by construction and were not assumed equal. Cut 3 / cut 5 and presence counts are diagnostics only; the main cut stays 4.')
    w('')
    # ------------------------------------------------------------------ 3 A selection
    w('## 3. A: independent T and N q pools (selection on cohort Q_SELECT, before TEST)')
    w('')
    w('Candidate pool, parameters and the rule (match-weighted Q_SELECT Brier, then log loss, then name) are the unchanged full-run ones '
      '(`fc20260915_fit_q.fit_candidates`); fits use all eligible cohort TRAIN rows (held-out-fold labels, p_pre_V), calibrators the cohort '
      'Q_CAL rows. Pooled reference = frozen full-run choice (h90 ridge_isotonic, h60 ridge_sigmoid, h120 ridge_isotonic).')
    w('')
    for h in (90, 60, 120):
        w(f'**h{h}** Q_SELECT Brier (chosen *):')
        w('')
        w('| candidate | ' + ' | '.join(K.COHORTS) + ' |')
        w('|---|---:|---:|')
        sels = {c: J(f'selection/q_specialist_{c}_h{h}.json') for c in K.COHORTS}
        for cand in C.Q_CANDIDATES:
            w(f'| {cand} | ' + ' | '.join(f'{sels[c]["select_metrics"][cand]["brier"]:.6f}{" *" if sels[c]["chosen"] == cand else ""}' for c in K.COHORTS) + ' |')
        w('| pooled frozen q on the same rows (diagnostic, not a candidate) | ' + ' | '.join(
            f'{sels[c]["pooled_reference"]["DIAGNOSTIC_cohort_validation_metrics_not_used_for_selection"]["select"]["brier"]:.6f}' for c in K.COHORTS) + ' |')
        w('| TRAIN / Q_CAL / Q_SELECT rows | ' + ' | '.join(
            f'{n(sels[c]["split_counts"]["train"]["rows"])} / {n(sels[c]["split_counts"]["calibrate"]["rows"])} / {n(sels[c]["split_counts"]["select"]["rows"])}' for c in K.COHORTS) + ' |')
        w('')
    # ------------------------------------------------------------------ 4 A evaluation
    w('## 4. A: pooled vs specialist on identical rows (frozen; equal match weights; paired match bootstrap 1000)')
    w('')
    w('Differences are specialist minus pooled (negative Brier / log loss = specialist better; positive AUC = specialist better). '
      'Rows of different cohorts are never compared as an improvement.')
    w('')
    for h in ('h90', 'h60', 'h120'):
        w(f'### {h}')
        w('')
        w('| Set | cohort | rows (matches) | spec. | pooled AUC / Brier / LL | specialist AUC / Brier / LL | dBrier [95% CI] | dLogLoss [95% CI] | dAUC [95% CI] | cal. slope pooled / spec. | ECE pooled / spec. | exact 0/1 (opp.) pooled / spec. |')
        w('|---|---|---:|---|---|---|---|---|---|---|---|---|')
        for s in SETS:
            r = RA[s][h]
            for coh, cc in r['cohorts'].items():
                b = pair(cc, 'specialist', 'pooled')
                P, S_ = cc['pooled'], cc['specialist']
                w(f'| {s} | {coh} | {n(cc["rows"])} ({n(cc["matches"])}) | {cc["specialist_chosen"]} | {f(P["auc"])} / {f(P["brier"])} / {f(P["logloss"])} | '
                  f'{f(S_["auc"])} / {f(S_["brier"])} / {f(S_["logloss"])} | {diff(b, "brier")} | {diff(b, "logloss")} | {diff(b, "auc", 4)} | '
                  f'{f(P["slope"], 3)} / {f(S_["slope"], 3)} | {f(P["ece_10bin"])} / {f(S_["ece_10bin"])} | {P["exact_0_or_1"]} ({P["exact_0_or_1_opposite_label"]}) / {S_["exact_0_or_1"]} ({S_["exact_0_or_1_opposite_label"]}) |')
        w('')
        w('Inside N (N specialist vs pooled on the same pick or skirmish rows):')
        w('')
        w('| Set | subset | rows | pooled Brier | N specialist Brier | dBrier [95% CI] | dLogLoss [95% CI] |')
        w('|---|---|---:|---:|---:|---|---|')
        for s in SETS:
            for lab, sub in RA[s][h]['cohorts']['N']['inside_N'].items():
                b = pair(sub, 'N_specialist', 'pooled')
                w(f'| {s} | {lab} | {n(sub["rows"])} | {f(sub["pooled"]["brier"])} | {f(sub["N_specialist"]["brier"])} | {diff(b, "brier")} | {diff(b, "logloss")} |')
        w('')
        w('Oracle-cohort routing (**membership needs post-cutoff participation; NOT a deployable live model**) vs pooled on all known rows:')
        w('')
        w('| Set | rows | routed Brier | pooled Brier | dBrier [95% CI] | dLogLoss [95% CI] | pooled E baseline reproduces full run |')
        w('|---|---:|---:|---:|---|---|---|')
        for s in SETS:
            o = RA[s][h]['oracle_routing_NOT_DEPLOYABLE']
            b = pair(o, 'oracle_routed', 'pooled')
            w(f'| {s} | {n(o["rows"])} | {f(o["routed"]["brier"])} | {f(o["pooled"]["brier"])} | {diff(b, "brier")} | {diff(b, "logloss")} | {RA[s][h]["E_pooled_all_valid_rows"]["reproduces_full_run_metrics"]} |')
        w('')
    # ------------------------------------------------------------------ 5 roles
    w('## 5. B: role supervision, meta-role classifier and team role posteriors')
    w('')
    w('**Supervision source (weak).** Original raw match-detail files for the main corpus are not on disk under the project-configured paths '
      '(core/config.py DETAIL_DIR default does not exist; the legacy path holds a later 15.18-15.22 collection with no main-corpus IDs; '
      'docs/DATA_AVAILABILITY.md states the cache is the sole copy). Per the specification, cached `role_slots` of 15.14 TRAIN matches are used only '
      'as an explicitly **weak** annotation: `core/roles.get_role_slots_from_detail` maps teamPosition (individualPosition fallback, which in practice '
      'never rescues an empty teamPosition because Riot reports "Invalid") and silently fills unmatched slots by participant ID. Teams whose slot '
      f'sequence in JSON key order reveals an ID fill are excluded: {n(prov["team_reasons"].get("detectable_participant_id_fill", 0))} of '
      f'{n(prov["train_teams"])} TRAIN teams excluded, {n(prov["teams_used"])} used ({n(prov["participants_used"])} participants). Whole-team or '
      'trailing-slot fills cannot be detected and remain. role_slots of VALIDATION/TEST/external matches were never read.')
    au = prov['available_raw_audit']['files']
    w(f'Available raw 15.14 detail outside the project (user Downloads, {len([x for x in au if not x.get("duplicate_of_previous")])} distinct TRAIN matches): '
      + '; '.join(f'{x["match"][:5]}... teamPosition equals cached slot role {x["teamPosition_equals_cached_role_slots"]}/{x["participants_n"]}, empty teamPosition {x["teamPosition_empty"]}'
                  for x in au if not x.get('duplicate_of_previous')) + '. Two matches cannot estimate a fallback rate.')
    w('')
    w('**Classifier (fixed, no tuning).** Inputs per participant: champion ID (one-hot, `handle_unknown=ignore`) and the two selected summoner spells '
      '(unordered multi-hot); multinomial LogisticRegression(C=1, lbfgs, max_iter 2000); one row per eligible participant, equal total weight per match. '
      'Final model on all eligible TRAIN teams, 5 OOF models on the existing TRAIN folds (TRAIN q rows use their own held-out fold model).')
    w('')
    w('| model | rows | fit matches | champion vocab | spell vocab | lbfgs iterations | converged |')
    w('|---|---:|---:|---:|---:|---:|---|')
    for k, v in rman['models'].items():
        fr = v['fit_record']
        w(f'| {k} | {n(fr["n_rows"])} | {n(fr["fit_matches"])} | {fr["champion_vocabulary_size"]} | {fr["spell_vocabulary_size"]} | {fr["n_iter"]} | {fr["converged"]} |')
    ag = rman['outputs']['MAIN_TRAIN']['OOF_agreement_with_WEAK_proxy_NOT_validated_accuracy']
    w('')
    w(f'OOF agreement with the **weak proxy itself** (not validated accuracy): posterior argmax {ag["posterior_argmax_agreement"]:.4f}, classifier argmax '
      f'{ag["classifier_argmax_agreement"]:.4f}, team exact assignment {ag["teams_exact_assignment"]:.4f}; per role '
      + ', '.join(f'{r} {v:.3f}' for r, v in ag['per_role_posterior_agreement'].items()) + '.')
    w('')
    w('**Team posterior.** 120 one-to-one assignments weighted by the product of participant role probabilities (clipped at 1e-12 inside logs), '
      'logsumexp-normalized and marginalized. This imposes one player per role as a modeling assumption; flex picks and lane swaps remain uncertain. '
      'Row and column sums were checked (max deviation ~1e-15); identical unseen drafts give uniform 0.2 marginals; permutation equivariance is tested.')
    w('')
    w('| Set | matches | mean assignment entropy (nats) | mean max marginal | unseen champion participants | teams with any unseen champion |')
    w('|---|---:|---:|---:|---:|---:|')
    for k in ('MAIN_TRAIN', 'MAIN_VALIDATION'):
        v = rman['outputs'][k]
        w(f'| {k} ({"OOF" if k == "MAIN_TRAIN" else "final"}) | {n(v["matches"])} | {v["mean_assignment_entropy"]:.4f} | {v["mean_max_marginal"]:.4f} | {v["champion_unseen_participant_rate"]:.4f} | - |')
    for k, v in RCj['role_outputs'].items():
        w(f'| {k} (final) | {n(v["matches"])} | {v["mean_assignment_entropy"]:.4f} | {v["mean_max_marginal"]:.4f} | {v["champion_unseen_participant_rate"]:.4f} | {v["teams_with_any_unseen_champion"]:.4f} |')
    w('')
    w('**Role reliability on independent annotations (post-freeze diagnostic).** External raw detail `teamPosition` (Riot position assignment, not '
      'observed spatial lane), teams with five distinct valid positions:')
    w('')
    w('| Set | participants | posterior argmax acc. | classifier argmax acc. | mean P(true role) | posterior log loss | team exact | seen / unseen champion acc. (n) | proxy (get_role_slots) = teamPosition on valid teams | invalid teams: detectable / undetectable fill |')
    w('|---|---:|---:|---:|---:|---:|---:|---|---:|---|')
    for s, v in rel['results'].items():
        pa = v['proxy_mechanism_audit']
        w(f'| {s} | {n(v["participants_evaluated"])} | {v["posterior_argmax_accuracy"]:.4f} | {v["classifier_argmax_accuracy"]:.4f} | {v["mean_posterior_prob_true_role"]:.4f} | '
          f'{v["posterior_log_loss"]:.4f} | {f(v["team_exact_assignment_rate"])} | {f(v["seen_champion"]["accuracy"])} ({n(v["seen_champion"]["participants"])}) / '
          f'{f(v["unseen_champion"]["accuracy"])} ({n(v["unseen_champion"]["participants"])}) | {pa["valid_teams_proxy_equals_teamPosition_participant_rate"]:.4f} | '
          f'{pa["invalid_teams_detectable_by_key_order"]} / {pa["invalid_teams_undetectable"]} |')
    w('')
    w('Per-role recall (posterior argmax vs teamPosition): ' + '; '.join(
        f'{s}: ' + ', '.join(f'{r} {x:.3f}' for r, x in v['per_role_recall'].items()) for s, v in rel['results'].items()) + '. Raw sha256 mismatches: '
      + ', '.join(f'{s} {v["raw_sha256_mismatch"]}' for s, v in rel['results'].items()) + '; draft vs raw champion/spell mismatches: '
      + ', '.join(f'{s} {v["draft_vs_raw_champion_or_spell_mismatch_matches"]}' for s, v in rel['results'].items()) + '. These are later patches/regions than '
      'TRAIN, so they measure transfer of the draft-to-role mapping, not the in-distribution accuracy on 15.14-15.16.')
    w('')
    # ------------------------------------------------------------------ 6 C
    w('## 6. C: controlled h90 representation comparison')
    w('')
    w('Arms: **participant** (frozen ridge set, 352 numeric: ridge C=.01 and a NEW full-feature LightGBM with the economic-model parameters), '
      '**draft control** (participant set + per-slot champion one-hot + unordered spell multi-hot, encoders fit on cohort TRAIN), **role** (191 global '
      'features + p_pre_V + role-weighted blue/red features 2x5x16 + role differences 5x16 + difference x time for gold/xp/level 5x3 + team role-uncertainty '
      '2x7 = 461; no participant-slot block, no direct champion/spell indicators). Feature sets differ; this is not a parameter-count-only comparison. '
      'TRAIN rows use OOF role posteriors. Each base has raw/sigmoid/isotonic variants.')
    w('')
    for coh in K.COHORTS:
        s = J(f'selection/arms_{coh}_h90.json')
        w(f'**{coh}** Q_SELECT Brier (within-arm winner *; overall choice **{s["overall_choice"]}**; ranking {s["overall_ranking"]}):')
        w('')
        w('| arm | features | ridge raw | ridge sigmoid | ridge isotonic | LGBM raw | LGBM sigmoid | LGBM isotonic | ridge fit s / iter | LGBM fit s |')
        w('|---|---:|---:|---:|---:|---:|---:|---:|---|---:|')
        for arm in K.ARMS:
            lg = s['fit_logs'][arm]
            cells = []
            for cand in K.arm_candidates(arm):
                cells.append(f'{s["select_metrics"][cand]["brier"]:.6f}{" *" if s["arm_winners"][arm] == cand else ""}')
            w(f'| {arm} | {lg["features"]} | ' + ' | '.join(cells) + f' | {lg["ridge_seconds"]} / {lg["ridge_n_iter"]} | {lg.get("lgbm_seconds")} |')
        w(f'| A specialist ({s["A_specialist_choice"]}) | | {s["A_specialist_select_metrics"]["brier"]:.6f} | | | | | | | |')
        w('')
    w('TEST and external paired comparisons (a minus b; fixed frozen models; the overall-vs-A pair mixes base model class and representation, so the '
      'controlled representation effect is read from the arm-winner pairs, which here share the LightGBM base and calibration family):')
    w('')
    w('| Set | cohort | rows | pair | dBrier [95% CI] | dLogLoss [95% CI] | dAUC [95% CI] |')
    w('|---|---|---:|---|---|---|---|')
    for s in SETS:
        for coh, cc in RC.get(s, {}).items():
            for p in (cc.get('bootstrap') or {}).get('pairs', []):
                if p['a'].startswith('drop_'):
                    continue
                w(f'| {s} | {coh} | {n(cc["rows"])} | {p["a"]} - {p["b"]} | {diff(p, "brier")} | {diff(p, "logloss")} | {diff(p, "auc", 4)} |')
    w('')
    w('Metrics of the chosen candidates (AUC / Brier / log loss / calibration slope / ECE):')
    w('')
    w('| Set | cohort | role winner | participant winner | draft winner | A specialist | pooled |')
    w('|---|---|---|---|---|---|---|')
    for s in SETS:
        for coh, cc in RC.get(s, {}).items():
            m, aw = cc['metrics'], cc['arm_winners']

            def mm(c):
                x = m[c]
                return f'{f(x["auc"])} / {f(x["brier"])} / {f(x["logloss"])} / {f(x["slope"], 3)} / {f(x["ece_10bin"])}'
            w(f'| {s} | {coh} | {mm(aw["role"])} | {mm(aw["participant"])} | {mm(aw["draft"])} | {mm("A_specialist")} | {mm("pooled")} |')
    w('')
    ab = J('selection/ablations_T_h90.json')
    w(f'**Leave-one-role-block-out ablations (T only).** Preselected full role ridge: {ab["preselected_full_role_ridge"]} (calibration '
      f'{ab["primary_calibration"]}); each block drop removes 51 columns (blue/red role features, differences, phase interactions) and refits the '
      'ridge and calibrators before TEST; global context remains, so information redundant with other roles or global totals can remain. '
      'This is a model ablation, not causal importance.')
    w('')
    w('| Set | dropped block | own Q_SELECT choice | Q_SELECT Brier (drop / full) | TEST-set dBrier [95% CI] (drop - full) | dLogLoss [95% CI] | dAUC [95% CI] |')
    w('|---|---|---|---|---|---|---|')
    for s in SETS:
        cc = RC.get(s, {}).get('T')
        if not cc:
            continue
        for g_, bv in ab['blocks'].items():
            p = pair(cc, f'drop_{g_}', 'full_role_ridge_preselected')
            w(f'| {s} | {g_} | {bv["own_q_select_choice"]} | {bv["select_metrics"][bv["primary_candidate"]]["brier"]:.6f} / {bv["full_role_ridge_select_metrics"]["brier"]:.6f} | '
              f'{diff(p, "brier")} | {diff(p, "logloss")} | {diff(p, "auc", 4)} |')
    w('')
    if diag:
        w('**Ridge optimizer-path diagnostic (post-freeze, TRAIN/VALIDATION only, not used for any choice).** The participant-arm ridge and the '
          'A-specialist ridge share rows, inputs, weights and C but were fit through different array paths with sklearn\'s default lbfgs tolerance (1e-4):')
        w('')
        w('| cohort | Q_SELECT abs(production A - production arm) mean / p99 / max | lbfgs iter (A / arm) | tight-tol refits: A path vs arm path max | production A vs tight max | production arm vs tight max | Q_SELECT Brier production A / arm / tight |')
        w('|---|---|---|---|---|---|---|')
        for coh, v in diag['cohorts'].items():
            t = v['tight_tolerance_refits']
            d0 = v['select_prediction_difference']
            w(f'| {coh} | {d0["mean_abs"]:.2e} / {d0["p99"]:.2e} / {d0["max"]:.2e} | {v["n_iter"]["A_specialist"]} / {v["n_iter"]["participant_arm"]} | '
              f'{t["tight_A_vs_tight_arm"]["max"]:.2e} | {t["production_A_vs_tight"]["max"]:.2e} | {t["production_arm_vs_tight"]["max"]:.2e} | '
              f'{t["select_brier"]["production_A"]:.6f} / {t["select_brier"]["production_arm"]:.6f} / {t["select_brier"]["tight"]:.6f} |')
        w('')
    # ------------------------------------------------------------------ 7 SHAP
    w('## 7. Role-arm explanations (exact 7-group Shapley on the final calibrated probability)')
    w('')
    w('Model = each cohort\'s frozen **role-arm winner** (this explains the role model whether or not it is the overall winner). Groups: five role '
      'groups (blue and red role features, differences and phase interactions of that role), global context (all non-participant state features, '
      'time, p_pre_V), role uncertainty; 128 coalitions; 256 hash-selected TEST rows per cohort, 128 hash-selected cohort TRAIN background rows '
      '(OOF role posteriors). Interventional replacement breaks derived relations and role ambiguity remains, so these are descriptive associations '
      'with generated labels, not causal lane importance (Lundberg & Lee 2017 framework).')
    w('')
    for coh, v in sh['cohorts'].items():
        c = v['checks']
        w(f'**{coh}: {v["model"]}** (overall h90 winner {v["overall_h90_winner"]}); {v["coverage_note"]}; background {v["background_rows"]} rows; '
          f'base value {v["base_value_mean"]:.4f}; additivity {c["additivity_max_abs"]:.1e}, phi+base = selected calibrated prediction {c["phi_plus_base_equals_selected_calibrated_prediction_max_abs"]:.1e}, '
          f'reload identical {c["reload_first3_bitwise_equal"]}, {c["model_evaluations"]:,} model evaluations in {c["runtime_s"]} s.')
        w('')
        w('| group | columns | mean abs phi [bootstrap 95% CI over explained rows] | mean abs (match-weighted) | mean signed | ' + ' | '.join(f'mean abs, start {b} min (n={d["n"]})' for b, d in v['time_bands'].items()) + ' |')
        w('|---|---:|---|---:|---:|' + '---:|' * len(v['time_bands']))
        for g_ in K.ROLE_GROUPS:
            ci = v['global_mean_abs_bootstrap_ci95'][g_]
            w(f'| {g_} | {v["group_sizes"][g_]} | {v["global_mean_abs"][g_]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {v["global_mean_abs_match_weighted"][g_]:.4f} | {v["global_mean_signed"][g_]:+.5f} | '
              + ' | '.join(f'{d["mean_abs"][g_]:.4f}' if d['mean_abs'] else 'NA' for d in v['time_bands'].values()) + ' |')
        w('')
        for lc in v['local_cases'][:2]:
            top = sorted(lc['group_shapley'].items(), key=lambda kv: -abs(kv[1]))[:3]
            teams = '; '.join(f'{t["team"]} (assignment entropy {t["assignment_entropy"]:.2f}): ' + ', '.join(
                f'{p["champion"] or p["champion_id"]} {p["estimated_role"]} {p["max_marginal"]:.2f}{" UNSEEN" if p["champion_unseen_in_role_vocabulary"] else ""}' for p in t['participants'])
                for t in lc['teams'])
            w(f'* Local case (start {lc["start_minute"]} min, Y={lc["y_h90"]}, q={lc["q_final"]:.4f}, base {lc["base_value"]:.4f}); largest signed groups: '
              + ', '.join(f'{k} {x:+.4f}' for k, x in top) + f'. Draft role estimates (champion, estimated role, max marginal): {teams}.')
        w('')
    w('All eight local cases per cohort (signed group values, per-participant role marginals and confidence) are in shap/shap_summary.json.')
    w('')
    # ------------------------------------------------------------------ 8 verification
    w('## 8. Verification record')
    w('')
    for k, v in sd.items():
        w(f'- [{"pass" if v else "FAIL"}] {k}')
    w('')
    # ------------------------------------------------------------------ 9 limitations
    w('## 9. Deviations, disclosures and limitations')
    w('')
    for x in [
        'Prior exposure: TEST 15.16 and all four external sets were evaluated for the pooled q (and V) in the completed full run; KR 16.13 also in earlier pilot work. Results are exploratory re-evaluations.',
        'Labels are generated by a fitted V (TRAIN: held-out-fold V); Y is estimated probability improvement, not a won fight or human judgement. Labels and V were not changed.',
        'Cohort membership (participation count) is post-cutoff information. Specialist-vs-pooled comparisons are within identical cohort rows; oracle routing is not deployable.',
        'Role supervision is a weak proxy (cached role_slots) because raw main detail is unavailable; undetectable participant-ID fills can remain. OOF agreement with that proxy is not accuracy. External teamPosition reliability is a post-freeze transfer diagnostic on later patches/regions.',
        'Role posteriors come only from champion and spells; they are estimated meta roles, not observed lanes. The one-player-per-role assignment is a modeling assumption; flex picks and swaps stay uncertain.',
        'Role weights express role uncertainty, not lane importance; no hand-set lane values or lane reward entered labels or models.',
        'The full-feature participant LightGBM is a new model; arm differences include base-model class effects where arms choose different bases (not the case for the chosen arm winners here, which are all LightGBM in each cohort).',
        'Ridge fits use sklearn default lbfgs tolerance (1e-4), as in the frozen full run (pooled ridge candidates included). Identical designs fit through different array paths differ (N: mean 8e-4, max 0.029 on Q_SELECT rows), and default-tolerance fits sit up to ~0.06 (single rows) from a tight-tolerance optimum while aggregate Q_SELECT Brier changes by < 2e-5 (section 6 diagnostic). The frozen models were not refit; small ridge-vs-ridge differences should be read with this optimizer noise in mind.',
        'Isotonic-calibrated candidates can output exact 0/1 probabilities (counts reported per cell); probabilities were not repaired using TEST.',
        'Bootstrap intervals cover test-sample (match) variability only; training, calibration, selection and role-model variability are not included. Many cells are reported; no multiplicity adjustment was made.',
        'SHAP covers 256 of the eligible TEST rows per cohort (coverage stated) with a 128-row background; group attributions are descriptive.',
        'Smoke runs used TRAIN rows only (1/8 match subset, pseudo Q_CAL/Q_SELECT from TRAIN folds). In the role-model smoke, pseudo-validation TRAIN rows received OOF (not final) posteriors; smoke results were not used for any choice.',
        'Implementation defects found and fixed before full fits: StateV2 participant slots are numbered 0-9; a forbidden-token check matched "UTILITY_"; a test fixture was wrong; ridge reload identity requires the same featurize-then-predict path; a Downloads timeline file was mistaken for detail. Fixes are logged in commands.txt.',
        'draft/draft_manifest.json describes slots as "1-5 Blue, 6-10 Red" in ordinal terms; array index and StateV2 names are 0-4 Blue, 5-9 Red.',
        'Atakhan/FEAT_UPDATE absence and unseen champions in 16.xx external sets (full-run report) also affect these models; role vocabulary unseen-champion rates are listed in section 5.',
        'The old manuscript was not edited. No file in the original repo or the worktree changed (integrity snapshot; mtime scan). Side effects in the workspace: '
        'new bytecode caches scripts/__pycache__/cr20260915_*.pyc, fc20260915_shap.cpython-313.pyc and tests/__pycache__/test_cr20260915_contracts*.pyc '
        '(pytest and in-process imports); runtime_repo/ under this output root. Nothing was deleted.',
    ]:
        w(f'- {x}')
    w('')
    w('Literature scope (as verified in the specification): Maymin (2021), Smart kills and worthless deaths, JQAS, DOI 10.1515/jqas-2019-0096 '
      '(tables 6/7: role-specific associations of performance bundles with match outcome) motivates role-aware analysis only, not this classifier, '
      'causal effects, the 90 s horizon or predicted gains. Lee & Ramler (2017), Identifying and Evaluating Successful Non-meta Strategies in League of '
      'Legends, FDG, DOI 10.1145/3102071.3102081, treats role/composition inference with spells and end-of-match items; end-of-match items are not '
      'allowed here and this classifier is not a reproduction. Lundberg & Lee (2017), arXiv:1705.07874, is the attribution framework; the grouping and '
      'background budget are operational choices. Cut 4 and the participation rule are the local v3.3 manifest/manuscript convention with reported '
      'cut 3/4/5 sensitivity, not a universal threshold from a paper.')
    w('')
    w('## 10. Artifacts')
    w('')
    w('protocol.json; status.json and status/; commands.txt; logs/; integrity/ (snapshots, diff); cohorts/ (cohort_manifest.json, <set>_cohort.npz, '
      'negative_count_provenance.json, detector_train_fixture.json); draft/ (draft npz, draft_manifest.json); role_supervision_provenance.json; '
      'contract_tests/; smoke_train_only/; selection/ (q_specialist_*, arms_*, ablations_T_h90, summaries); q_fit_metrics/; predictions/ (specialist '
      'trainval); models/q_specialist, models/arms, models/ablations; role_models/ + role_models_manifest.json; role_outputs/; frozen_manifest.json; '
      'label_access_log.jsonl; eval/ (results_A.json, results_C.json, role_reliability_external_raw.json, predictions/, hashes_*.json); shap/; '
      'diagnostics/; validation.json; DEFINITION_AND_EVIDENCE.md. Scripts: scripts/cr20260915_*.py; tests: tests/test_cr20260915_contracts.py.')
    (O / 'REPORT.md').write_bytes('\n'.join(L).encode('utf-8'))
    definitions(cm, prov, fz, rman, rel, sh, val)
    K.Status('report').update('complete', 'report', report_sha256=C.sha256_file(O / 'REPORT.md'),
                              definitions_sha256=C.sha256_file(O / 'DEFINITION_AND_EVIDENCE.md'),
                              validation_failed=val['failed'] if val else None, next_step='none (task complete)')
    print('report written', len(L), 'lines')


def definitions(cm, prov, fz, rman, rel, sh, val):
    L = []
    w = L.append
    w('# Definitions and evidence: cohort / role-aware q experiment (2026-09-15)')
    w('')
    w('Each entry: definition, status (fixed input / specification rule / implementation choice / empirical evidence) and where the evidence is.')
    w('')
    rows = [
        ('Engagement population E', 'every detected engagement row of the frozen full-corpus labels (566,452 main; 54,686 external); comparisons use rows valid at the horizon', 'fixed input', 'cohorts/cohort_manifest.json join'),
        ('Participation count', 'stored v3.3 det_cluster_blue/red: kill participants plus in-radius (3,000 u) interaction actors incl. shop events; merged engagements keep the earlier candidate counts', 'fixed input (v3.3 detector)', 'manuscript sec_definition.tex scale subsection; gameplay/fights.py; detector_train_fixture.json'),
        ('Negative count', '-1 = int(0 or -1); read as 0 only when re-detection proves a raw integer 0 at the same (s, L)', 'specification rule; empirical proof', 'cohorts/negative_count_provenance.json'),
        ('Teamfight T', 'min(cluster_blue, cluster_red) >= 4 (manifest scale.teamfight_min)', 'fixed convention (v3.3), not a universal threshold', 'corpus_shards_v33/manifest.json'),
        ('Non-teamfight N', 'known scale and min < 4; fine classes pick (<= 1) and skirmish (2..3) are diagnostics', 'specification rule', 'cohort_manifest.json'),
        ('Unknown scale', 'negative count not proven zero; excluded from T/N, kept in E (none observed)', 'specification rule', 'cohort_manifest.json'),
        ('Label Y_h', '1[V(endpoint_h) - V(s - 1 ms) > 0], same adapter; TRAIN held-out-fold V, others final V; h90 primary', 'fixed input (frozen full run)', 'outputs/full_corpus_training_20260915/labels'),
        ('q', 'probability of generated Y from pre-only inputs (StateV2 at q_pre minus snapshot age, plus p_pre_V)', 'fixed input schema', 'full run q_pre_only_schema.json'),
        ('Pooled q', 'frozen full-run choice per horizon, fit on E', 'fixed reference', 'frozen_manifest.json pooled_reference'),
        ('Specialist q', 'unchanged candidate pool fit on cohort TRAIN rows, calibrated on cohort Q_CAL, chosen on cohort Q_SELECT', 'specification rule', 'selection/q_specialist_*.json'),
        ('Oracle-cohort routing', 'specialist chosen by the post-cutoff cohort label; not deployable', 'reporting device only', 'eval/results_A.json'),
        ('Weak role annotation', 'cached role_slots of eligible 15.14 TRAIN teams (teamPosition-derived with silent ID fill); detectable fills excluded', 'specification fallback clause (raw main detail unavailable)', 'role_supervision_provenance.json'),
        ('Meta-role classifier', 'champion one-hot + unordered spell multi-hot -> multinomial logistic regression C=1', 'implementation choice fixed by the specification', 'role_models_manifest.json'),
        ('Team role posterior W', 'marginals over 120 one-to-one assignments with weights prod p_i(role)', 'modeling assumption (one player per role)', 'cr20260915_common.team_role_posterior; tests'),
        ('Role uncertainty summaries', 'per team: assignment entropy, mean/min max marginal, champion missing/unseen counts, spell missing/unseen counts', 'implementation choice', 'protocol.json interpretation_choices'),
        ('Role representation', 'global features + role-weighted blue/red participant features, blue-minus-red differences, difference x time_minutes (gold/xp/level), uncertainty summaries', 'specification rule', 'cr20260915_common.build_role_matrix'),
        ('Draft control', 'participant representation + per-slot champion one-hot + per-slot unordered spell multi-hot (cohort TRAIN encoders)', 'specification rule; unscaled indicators are an implementation choice', 'selection/arms_*_h90.json'),
        ('Full-feature participant LightGBM', 'economic-model parameters on all 352 numeric pre inputs', 'new model (identified as such)', 'selection/arms_*_h90.json'),
        ('Role-block ablation', 'role ridge refit without one role group (51 columns), same calibration family as the preselected full role ridge', 'model ablation, not causal importance', 'selection/ablations_T_h90.json'),
        ('Group Shapley', 'exact interventional Shapley over 7 groups on the final calibrated probability; TRAIN background', 'descriptive explanation', 'shap/shap_summary.json'),
        ('Role reliability', 'final role model vs external raw teamPosition (post-freeze)', 'empirical evidence (transfer to later patches)', 'eval/role_reliability_external_raw.json'),
    ]
    w('| Term | Definition | Status | Evidence |')
    w('|---|---|---|---|')
    for r in rows:
        w('| ' + ' | '.join(r) + ' |')
    w('')
    w('## Hypotheses versus evidence')
    w('')
    w('- Hypothesis: separate T and N q pools capture scale-specific structure. Evidence: section 4 of REPORT.md (paired, identical rows). Cohort choice itself is not a live input.')
    w('- Hypothesis: role-organized participant information adds predictive signal beyond slot order and beyond direct draft indicators. Evidence: section 6 arm-winner pairs (controlled within cohort and base) and T ablations; redundancy with global totals is expected.')
    w('- Not claimed: causal lane importance, validated role accuracy on the main corpus, a deployable routing model, or a confirmatory TEST result.')
    w('')
    w(f'Verification: validation.json ({len(val["checks"])} checks, failed: {val["failed"] or "none"}).')
    (K.OUT / 'DEFINITION_AND_EVIDENCE.md').write_bytes('\n'.join(L).encode('utf-8'))


if __name__ == '__main__':
    main()
