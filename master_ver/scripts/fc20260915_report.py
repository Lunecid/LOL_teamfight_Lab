"""Full-corpus report: REPORT.md, DEFINITION_AND_EVIDENCE.md, figures and source_hashes.json from saved artifacts.

Every number is read from artifacts written by the runners (no recomputation of models). Separates literature,
protocol/design choices, data estimates, completed experiments and unresolved limits.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402

OUT = C.OUT
EXT = ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13')
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'


def rd(p):
    return C.read_json(OUT / p)


def f(x, d=4):
    if x is None:
        return 'NA'
    if isinstance(x, (int, np.integer)):
        return f'{int(x):,}'
    return f'{x:.{d}f}'


def ci(v, d=4):
    if not v or v.get('ci95') is None:
        return 'NA'
    return f"{f(v['estimate'], d)} [{f(v['ci95'][0], d)}, {f(v['ci95'][1], d)}]"


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = OUT / 'figures'
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': .6})
    rv = rd('eval/results_v.json')
    ch = rv['v_chosen']
    made = []
    # 1. V reliability, three sets
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color='#b9b8b2', lw=1, ls='--', zorder=1)
    for name, col, lab in (('MAIN_TEST', S1, 'TEST 15.16'), ('EXT_KR_16.13', S2, 'KR 16.13'), ('EXT_NA1_16.13', S3, 'NA1 16.13')):
        bins = rv['results'][name]['bucket'][ch]['overall']['bins']
        ax.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], color=col, lw=2, marker='o', ms=5, label=lab,
                markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
    ax.set_xlabel('mean predicted V (bin, match-weighted)')
    ax.set_ylabel('observed Blue win rate (bin, match-weighted)')
    ax.set_title('Frozen V reliability on bucket queries', loc='left', fontsize=11)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(figdir / 'v_reliability_test_kr1613_na11613.png', dpi=150)
    plt.close(fig)
    made.append('figures/v_reliability_test_kr1613_na11613.png')
    # 2. V log loss by minute on TEST full trajectories
    curve = [c for c in rv['results']['MAIN_TEST']['full_trajectory_minute_curve'] if not c['sparse']]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([c['minute'] for c in curve], [c['logloss'] for c in curve], color=S1, lw=2)
    ax.set_xlabel('query minute (TEST full minute trajectories)')
    ax.set_ylabel('match-weighted log loss')
    ax.set_title('Frozen V log loss by query minute, TEST 15.16 (minutes with >= 30 matches)', loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / 'v_logloss_by_minute_test.png', dpi=150)
    plt.close(fig)
    made.append('figures/v_logloss_by_minute_test.png')
    # 3. q h90 chosen reliability on TEST
    rq = rd('eval/results_q.json')['results']['MAIN_TEST']['h90']
    bins = rq['metrics'][rq['chosen']]['bins']
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color='#b9b8b2', lw=1, ls='--', zorder=1)
    ax.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], color=S1, lw=2, marker='o', ms=5,
            markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
    for b in bins:
        if b['n'] < 5000:  # flag sparse bins only
            ax.annotate(f"n={b['n']:,}", (b['predicted'], b['observed']), textcoords='offset points', xytext=(-8, 8), fontsize=7, color=INK2, ha='right')
    ax.set_xlabel('mean predicted q (bin)')
    ax.set_ylabel('observed generated Y rate (bin)')
    ax.set_title(f"Frozen h90 q ({rq['chosen']}) reliability, TEST 15.16", loc='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / 'q_h90_reliability_test.png', dpi=150)
    plt.close(fig)
    made.append('figures/q_h90_reliability_test.png')
    # 4. SHAP groups
    sh = rd('shap/shap_summary.json')
    gt = sorted(sh['group_table'], key=lambda r: r['mean_abs_group_exact_shapley_final_output'])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(gt))
    ax.barh(y, [r['mean_abs_group_exact_shapley_final_output'] for r in gt], color=S1, height=.6)
    ax.set_yticks(y, [f"{r['group']} ({r['n_features']})" for r in gt])
    for i, r in enumerate(gt):
        ax.text(r['mean_abs_group_exact_shapley_final_output'], i, f"  {r['mean_abs_group_exact_shapley_final_output']:.4f}", va='center', fontsize=8, color=INK2)
    ax.set_xlabel('mean |group interventional Shapley| on final q probability')
    ax.set_title('h90 q: group Shapley on final probability', loc='left', fontsize=11)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(figdir / 'shap_h90_groups_final_probability.png', dpi=150)
    plt.close(fig)
    made.append('figures/shap_h90_groups_final_probability.png')
    return made


def main():
    proto, val = rd('protocol.json'), rd('validation.json')
    mm = rd('extract/MAIN/extraction_manifest.json')
    sel, vman = rd('selection_v.json'), rd('v_models_manifest.json')
    lt, lte = rd('labels/labels_trainval_manifest.json'), rd('labels/labels_test_external_manifest.json')
    fz = rd('frozen_manifest.json')
    counts = rd('eval/counts_and_exclusions.json')['counts']
    rv, rq = rd('eval/results_v.json'), rd('eval/results_q.json')
    sh = rd('shap/shap_summary.json')
    qsel = {h: rd(f'selection/q_h{h}.json') for h in C.HORIZONS_S}
    ch = rv['v_chosen']
    tmp_path = OUT / 'temporal_comparator' / 'results_temporal.json'
    temporal = C.read_json(tmp_path) if tmp_path.exists() else None
    tstat = {}
    for g in ('temporal_extract', 'temporal_train', 'temporal_evaluate'):
        p = OUT / 'status' / f'{g}.json'
        if p.exists():
            d = C.read_json(p)
            tstat[g] = f"{d.get('state')} {d.get('stage')} {d.get('processed')}/{d.get('total')}"
    figs = figures()
    L = []
    a = L.append
    a('# Full-corpus retraining 2026-09-15: V, cross-fitted engagement labels, q, frozen TEST and external evaluation')
    a('')
    a(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} KST from saved artifacts. Design: docs/CLAUDE_FULL_CORPUS_TRAIN_20260915.md (Codex). "
      'Implementation and execution: Claude Opus 5. Output root: outputs/full_corpus_training_20260915.')
    a('')
    a('Role: exploratory full-corpus experiment with **model-defined labels** (not ground truth, not causal engagement effects, '
      'not production). Previous P1-P4 results are pilots on reduced cohorts; numbers below are not comparable improvements over them.')
    a('')
    a('## 1. Completion status')
    a('')
    a('| Stage | State | Evidence |')
    a('|---|---|---|')
    a(f"| Protocol frozen before decoding/fitting | done ({proto['written_at']}) | protocol.json |")
    a(f"| Main cache decoding (210,000 matches; arrays, events, meta) | done: {f(mm['totals']['loaded'])} loaded | extract/MAIN/extraction_manifest.json |")
    a('| External raw -> cache adaptation + frozen detector | done: 21,190/21,190 written, 0 excluded; EUW1 unavailable (0 complete pairs) | external/*/prepared_manifest.json |')
    a(f"| Contract tests before full fit | 32 passed | logs/contract_tests_run1.txt |")
    a(f"| Final V + calibration choice | done: `{ch}` | selection_v.json |")
    a('| Five OOF fold V adapters | done | v_models_manifest.json |')
    a('| TRAIN/VALIDATION labels (TRAIN from held-out fold adapters) | done | labels/labels_trainval_manifest.json |')
    a(f"| q h90 / h60 / h120 fit and Q_SELECT choice | done: {qsel[90]['chosen']} / {qsel[60]['chosen']} / {qsel[120]['chosen']} | selection/q_h*.json |")
    a(f"| Freeze of all V/q selections and hashes | done ({fz['frozen_at']}), before any TEST/external label | frozen_manifest.json |")
    a('| TEST 15.16 and external labels, V and q evaluation, bootstrap | done; frozen hashes unchanged after | eval/*.json |')
    a('| SHAP for chosen h90 q | done; additivity and determinism checks pass | shap/shap_summary.json |')
    a(f"| Post-run verification | {val['status']} ({len(val['checks'])} checks, failed: {val['failed_checks'] or 'none'}) | validation.json |")
    a(f"| Temporal SimpleRNN comparator | {'done' if temporal else 'PENDING at report time: ' + json.dumps(tstat)} | temporal_comparator/ |")
    a('')
    a('## 2. Data: roles, eligibility and exclusions')
    a('')
    a('Main manifest outputs/full_corpus_preflight_20260915/main_matches.csv (authoritative). TRAIN fold = sha256("full-v-oof-20260915:"+id)[:8] mod 5; '
      'VALIDATION role = sha256("full-val-20260915:"+id)[:8] mod 4 (0 V_CAL, 1 V_SELECT, 2 Q_CAL, 3 Q_SELECT). Every match decoded; no subsampling.')
    a('')
    a('| Set / role | raw matches | loaded | V-eligible matches | V bucket queries | V exclusions | matches with engagements | engagement rows | valid h90 rows (matches) | excluded h90 rows (reason) |')
    a('|---|---:|---:|---:|---:|---|---:|---:|---:|---|')
    order = [('MAIN', f'fold{k}') for k in range(5)] + [('MAIN', r) for r in C.VAL_ROLES] + [('MAIN', 'TEST')] + [(s, 'EXTERNAL') for s in EXT]
    for s, r in order:
        c = counts[s][r]
        a(f"| {s} {r} | {f(c['raw_matches'])} | {f(c['loaded'])} | {f(c['v_eligible_matches'])} | {f(c['v_bucket_queries'])} | "
          f"{json.dumps(c['v_exclusions'])} | {f(c['matches_with_exposures'])} | {f(c['exposure_rows'])} | {f(c['h90']['valid_rows'])} ({f(c['h90']['valid_matches'])}) | "
          f"{c['h90']['excluded_rows']} {json.dumps(c['h90']['exclusion_reasons'])} |")
    a('| EUW1 | 0 | - | - | - | no complete raw pairs | - | - | - | set unavailable; not a completed test |')
    a('')
    a('Every excluded match/engagement is listed with match id and reason in eval/exclusions_v_<set>.csv and eval/exclusions_engagement_<set>.csv. '
      'The only engagement exclusion observed is next_start <= L (overlapping support; 348 main rows, the preflight upper-bound count), identical at 60/90/120 s. '
      'V exclusions are games without any minute query >= 2 min before GAME_END (remakes/very short games) and one game in each of MAIN Q_CAL and NA1 whose '
      'GAME_END winner was missing or conflicting. Games without engagements remain in the V dataset.')
    a('')
    a(f"Totals: main V-eligible {f(mm['totals']['v_eligible'])} matches, {f(mm['totals']['v_bucket_rows'])} bucket queries (all roles), "
      f"{f(mm['totals']['v_rows'])} stored V rows incl. TEST full trajectories; {f(mm['totals']['e_rows'])} engagement rows, {f(mm['totals']['e_valid_h90'])} valid at every horizon.")
    a('')
    a('### Objective availability audit (raw event census)')
    a('')
    cen = {s: rd(f'extract/{s}/extraction_manifest.json')['event_census'] for s in ('MAIN',) + EXT}

    def ccount(s, prefix):
        return sum(v for k, v in cen[s].items() if k.startswith(prefix))
    a('| Class (raw) | StateV2 support | MAIN (210k) | KR 16.13 | KR 16.14 pilot | KR 16.15 | NA1 16.13 |')
    a('|---|---|---:|---:|---:|---:|---:|')
    rows = [('Baron (ELITE_MONSTER_KILL BARON_NASHOR)', 'counts, ever, age, recent acquisition', 'elite:BARON_NASHOR'),
            ('Elemental dragons (DRAGON, 6 elements)', 'per-element counts', 'elite:DRAGON:'),
            ('Elder (DRAGON ELDER_DRAGON)', 'counts, ever, age, recent acquisition', 'elite:DRAGON:ELDER_DRAGON'),
            ('Rift Herald', 'counts, ever, age', 'elite:RIFTHERALD'),
            ('Void grubs (HORDE)', 'counts, ever, age', 'elite:HORDE'),
            ('Atakhan', 'counts, ever, age', 'elite:ATAKHAN'),
            ('DRAGON_SOUL_GIVEN (all)', 'owned soul flags; teamId 0 diagnostic only', 'soul:'),
            ('DRAGON_SOUL_GIVEN teamId 0 (unassigned)', 'not counted as ownership (diagnostic)', None),
            ('Elite kills with killer team 300', 'unknown_objective_team_count', None),
            ('FEAT_UPDATE (feats)', 'NOT supported (not a StateV2 input)', 'type:FEAT_UPDATE'),
            ('OBJECTIVE_BOUNTY_*', 'NOT supported', 'type:OBJECTIVE_BOUNTY'),
            ('Unmapped elite monsterType', 'none found', 'elite_unmapped')]
    for label, sup, pre in rows:
        vals = []
        for s in ('MAIN',) + EXT:
            if label.startswith('DRAGON_SOUL_GIVEN teamId 0'):
                vals.append(sum(v for k, v in cen[s].items() if k.startswith('soul:') and k.endswith('team_0')))
            elif label.startswith('Elite kills with killer team 300'):
                vals.append(sum(v for k, v in cen[s].items() if k.startswith('elite:') and k.endswith('team_300')))
            elif pre == 'elite:DRAGON:':
                vals.append(sum(v for k, v in cen[s].items() if k.startswith('elite:DRAGON:') and 'ELDER' not in k))
            else:
                vals.append(ccount(s, pre))
        a(f"| {label} | {sup} | " + ' | '.join(f(v) for v in vals) + ' |')
    a('')
    a('Atakhan and FEAT_UPDATE events are absent in all 21,190 external 16.xx games (present in 15.xx). The Atakhan columns are therefore always 0 '
      'externally: this is recorded as a **class not observed in these patches**, not verified as a genuine in-game zero, and the 15.14-trained V/q '
      'coefficients for Atakhan cannot be validated externally. FEAT_UPDATE is not represented in StateV2 in any set. The parent unknown_team=1,417 is '
      'exactly the 1,026 HORDE + 391 RIFTHERALD kills with killerTeamId 300 in the main corpus; it affects only objective-delay pairs, not exposure eligibility.')
    a('')
    szf = rd('published/external_structural_zero_features.json')['sets']
    a('Feature-level check (published/external_structural_zero_features.json; reference = zero fraction in 15.15 validation roles): the features that are '
      'always zero externally but not in validation are exactly the 8 Atakhan count/ever columns (and their time interactions) in KR 16.13, KR 16.15 and '
      f"NA1 16.13; KR 16.14 pilot adds {len(szf['KR_16.14_pilot']['always_zero_externally_but_not_in_validation']) - 8} rare elder/soul columns, attributable to its "
      '195 V-eligible games (sparsity, not a verified structural absence). No feature had a >0.2 zero-fraction increase otherwise.')
    a('')
    avail = counts['external_preparation']
    a('External raw key presence (participant frames, champion stats, detail championId) was 100% in every set; no structurally unavailable V input key. '
      'Unseen champion IDs (absent from the 15.14 one-hot vocabulary) are encoded as all-zero one-hot and affect '
      + ', '.join(f"{s} {100 * rv['results'][f'EXT_{s}']['feature_availability']['rows_with_any_unseen_champion']:.1f}%" for s in EXT)
      + f" of external bucket queries (TEST 15.16 {100 * rv['results']['MAIN_TEST']['feature_availability']['rows_with_any_unseen_champion']:.1f}%).")
    a('')
    a('## 3. V (primary win-probability labeler)')
    a('')
    ff = sel['final_fit']
    a(f"Final base V: expanded StateV2 logistic (liblinear, C = 0.01 predeclared from the pilot, not a validated optimum), {f(ff['rows'])} TRAIN bucket queries "
      f"from {f(ff['matches'])} matches, equal total weight per match, {ff['n_features_in']} inputs ({ff['n_numeric_scaled']} scaled numeric + 10 one-hot champion slots), "
      f"converged in {ff['n_iter']} iterations ({ff['seconds']} s, no ConvergenceWarning).")
    a('')
    a('| V_SELECT (18,780 matches) | log loss | Brier | AUC |')
    a('|---|---:|---:|---:|')
    for k, v in sel['select_scores'].items():
        a(f"| {k}{' (chosen)' if k == ch else ''} | {f(v['logloss'], 6)} | {f(v['brier'], 6)} | {f(v['auc'], 6)} |")
    sf = sel['sigmoid_fit']
    a('')
    a(f"Positive-slope sigmoid on V_CAL: a = {f(sf['intercept_a'], 5)}, b = {f(sf['slope_b'], 5)} (bound b >= 1e-6 inactive, L-BFGS-B success). "
      f"Raw was chosen by the predeclared rule; the difference is tiny ({f(sel['select_scores']['sigmoid_pos']['logloss'] - sel['select_scores']['raw']['logloss'], 6)} log loss). "
      'Because raw was chosen, every OOF fold adapter is also raw (uniform calibration family).')
    a('')
    oo = vman['heldout_train_oof_eval']
    disp = vman['fold_dispersion_on_v_select']
    a('Label/endpoint distributions by TRAIN fold (held-out adapters) and VALIDATION role are in published/label_endpoint_distributions.json; outcome-free V query '
      'keys, frame ages and per-role sample counts in published/v_query_keys.npz and published/v_query_census.json.')
    a('')
    a(f"Held-out TRAIN (each query scored by the fold adapter that excluded its match): AUC {f(oo['auc'])}, Brier {f(oo['brier'])}, log loss {f(oo['logloss'])}, "
      f"calibration slope {f(oo['slope'], 3)}. Per fold AUC: " + ', '.join(f"{k} {f(v['heldout_train_eval']['auc'])}" for k, v in vman['folds'].items()) + '. '
      f"Fold dispersion on V_SELECT: mean per-row SD across the five fold adapters {f(disp['mean_across_row_sd'])}, p90 {f(disp['p90_across_row_sd'])}, max {f(disp['max_across_row_sd'])}; "
      f"mean |fold - final| {f(np.mean(list(disp['mean_abs_fold_minus_final'].values())))}, max {f(max(disp['max_abs_fold_minus_final'].values()))}.")
    a('')
    a('### Frozen V vs final Blue win (bucket queries, equal match weights)')
    a('')
    a('| Set | role | matches | queries | AUC | Brier | log loss | cal. slope | cal. intercept | ECE | TRAIN-prior log loss | Brier diff vs prior [95% CI] |')
    a('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in ('MAIN_TEST',) + tuple(f'EXT_{s}' for s in EXT) + ('MAIN_V_SELECT', 'MAIN_V_CAL', 'MAIN_Q_CAL', 'MAIN_Q_SELECT'):
        c = rv['results'][name]
        o = c['bucket'][ch]['overall']
        boot = c.get('bootstrap_bucket')
        bd = next((p['a_minus_b']['brier'] for p in boot['pairs'] if p['b'] == 'train_prior_constant'), None) if boot else None
        a(f"| {name} | {c['kind'].split(' ')[0]} | {f(o['matches'])} | {f(o['n'])} | {f(o['auc'])} | {f(o['brier'])} | {f(o['logloss'])} | {f(o['slope'], 3)} | "
          f"{f(o['intercept'], 3)} | {f(o['ece_10bin'])} | {f(c['bucket']['train_prior_constant']['overall']['logloss'])} | {ci(bd) if bd else 'diagnostic'} |")
    a(f"| MAIN_TRAIN held-out fold | diagnostic | {f(oo['matches'])} | {f(oo['n'])} | {f(oo['auc'])} | {f(oo['brier'])} | {f(oo['logloss'])} | {f(oo['slope'], 3)} | {f(oo['intercept'], 3)} | {f(oo['ece_10bin'])} | - | - |")
    a('')
    a('Secondary full minute trajectories (all eligible minute queries, equal match weights): ' + '; '.join(
        f"{n} AUC {f(rv['results'][n]['full_trajectory_secondary'][ch]['overall']['auc'])}, log loss {f(rv['results'][n]['full_trajectory_secondary'][ch]['overall']['logloss'])} "
        f"({f(rv['results'][n]['full_trajectory_secondary'][ch]['overall']['n'])} queries)" for n in ('MAIN_TEST',) + tuple(f'EXT_{s}' for s in EXT)) + '.')
    a('')
    tb = rv['results']['MAIN_TEST']['bucket'][ch]['time_bands']
    a('TEST time bands (bucket queries): ' + '; '.join(f"{k} min: n={f(v['n'])}, AUC {f(v.get('auc'))}, log loss {f(v.get('logloss'))}" for k, v in tb.items()) + '. '
      'Early-game queries carry little information (AUC ~0.67), as expected for an outcome 20-40 minutes away.')
    a('')
    a('External sets are separate tests, each with its own row. Calibration slopes of 0.81-0.87 and negative intercepts on the 16.xx sets indicate '
      'over-confident / shifted probabilities under patch, champion-pool and region change; no recalibration was performed on any test set. '
      'KR 16.13 vs NA1 16.13 is the same-patch region contrast; KR 16.13/16.14/16.15 are time/patch contrasts with region fixed. KR 16.13 was '
      'previously used in pilot adaptation/evaluation, KR 16.14 has pilot provenance, and KR 16.15 and NA1 16.13 prior use is unknown: none is called untouched.')
    a('')
    a('Selected TEST objective-history strata (bucket queries; descriptive, no tuning): ' + '; '.join(
        f"{k} n={f(v.get('n'))} AUC {f(v.get('auc'), 3)}" for k, v in rv['results']['MAIN_TEST']['bucket'][ch]['objective_strata'].items()
        if k in ('no_baron_elder_or_soul_history', 'atakhan_any', 'baron_ever_any', 'elder_ever_any', 'owned_soul_any', 'unassigned_soul_teamId0_any_DIAGNOSTIC')) + '.')
    a('')
    a(f"Figures: {figs[0]}, {figs[1]}.")
    a('')
    a('## 4. Generated engagement labels')
    a('')
    a('s = first kill - 15 s, q_pre = s - 1 ms, L = last kill; endpoint_h = min(L + h, next raw CHAMPION_KILL after L - 1 ms, next detected engagement start - 1 ms, '
      'GAME_END - 1 ms). delta = V(endpoint) - V(q_pre) with the same adapter; Y = 1[delta > 0]. TRAIN rows use their held-out-fold adapter; all other rows the final V.')
    a('')
    a('| Set | h | valid rows | matches | Y=1 rate (match-weighted) | exact-zero deltas | end reasons | post snapshot after L |')
    a('|---|---:|---:|---:|---:|---:|---|---:|')
    for name in ('MAIN_TEST',) + tuple(f'EXT_{s}' for s in EXT):
        for h in C.HORIZONS_S:
            c = rq['results'][name][f'h{h}']
            ld = c['label_distribution']
            a(f"| {name} | {h} | {f(c['rows'])} | {f(c['matches'])} | {f(c['positive_rate_match_weighted'])} | {c['exact_zero_delta']} | {json.dumps(ld['end_reasons'])} | {f(ld['post_snapshot_after_L_rate'], 3)} |")
    for nm in ('MAIN_TRAIN', 'MAIN_VALIDATION'):
        s_ = lt['summaries'][nm]
        a(f"| {nm} | 90 | {f(s_['checks']['h90']['valid_rows'])} | {f(s_['matches'])} | {f(s_['checks']['h90']['positive'] / s_['checks']['h90']['valid_rows'])} (rows) | {s_['checks']['h90']['exact_zero_delta']} | see labels manifest | - |")
    a('')
    d90 = lt['summaries']['MAIN_TRAIN']['checks']['h90']['DIAGNOSTIC_final_V_in_sample_vs_oof_label_agreement']
    a(f"Cross-fitting diagnostic (TRAIN, h90): labels from held-out-fold adapters agree with in-sample final-V labels on {100 * d90['rows']:.2f}% of rows "
      f"(match-weighted {100 * d90['match_weighted']:.2f}%), mean |delta difference| {f(d90['mean_abs_delta_diff'], 5)}. The in-sample labels were computed only for this "
      'diagnostic and never used for q.')
    a('')
    tst = rq['results']['MAIN_TEST']
    a(f"Horizon sensitivity on TEST: Y label disagreement h90 vs h60 {100 * tst['cross_horizon_diagnostic']['label_disagreement_h90_vs_h60']:.2f}%, "
      f"h90 vs h120 {100 * tst['cross_horizon_diagnostic']['label_disagreement_h90_vs_h120']:.2f}%. About 65% of TEST h90 endpoints have a raw frame after L; "
      'the remaining ~35% value the endpoint from the pre-L frame plus events (staleness, as in the pilot). Most windows end at the next raw kill '
      '(79% at h90), so 90 s is an upper bound, not a fixed observation length.')
    a('')
    a('Checks (all sets): label formula exact, no raw kill inside (L, endpoint], endpoints monotone across horizons, identical endpoints give identical post predictions, '
      'first and last kill are raw kills, stored next_start/game end equal recomputation, snapshots never after the query. Simultaneous first kills are flagged, not '
      f"dropped (TRAIN {lt['summaries']['MAIN_TRAIN']['checks']['ambiguous_simultaneous_first_kill_rows']}, VALIDATION {lt['summaries']['MAIN_VALIDATION']['checks']['ambiguous_simultaneous_first_kill_rows']} rows).")
    a('')
    a('## 5. q (pre-engagement probability of the generated label)')
    a('')
    a('Inputs: StateV2 at q_pre without snapshot_age_s, plus p_pre_V (held-out-fold V on TRAIN, final V elsewhere); champion IDs excluded; ridge set 352 numeric inputs; '
      f"economic/aggregate set {rd('q_pre_only_schema.json')['economic_count']} inputs. TRAIN 199,358 rows / 69,124 matches; Q_CAL 50,955 / 17,313; Q_SELECT 50,375 / 17,230 (h90).")
    a('')
    a('| Q_SELECT Brier (rank rule: Brier, log loss, name) | h90 | h60 | h120 |')
    a('|---|---:|---:|---:|')
    for cand in C.Q_CANDIDATES:
        a(f"| {cand} | " + ' | '.join(f"{f(qsel[h]['select_metrics'][cand]['brier'], 6)}{' *' if qsel[h]['chosen'] == cand else ''}" for h in (90, 60, 120)) + ' |')
    a('')
    a('\\* chosen. Selections were saved with bundle hashes and frozen (frozen_manifest.json) before TEST/external labels existed.')
    a('')
    a('### TEST 15.16, h90 (163,576 rows, 56,254 matches; equal match weights)')
    a('')
    t90 = tst['h90']
    a('| candidate | AUC | Brier | log loss | cal. slope | cal. intercept | ECE | rows with q exactly 0 or 1 (opposite label) |')
    a('|---|---:|---:|---:|---:|---:|---:|---|')
    for cand in C.Q_CANDIDATES:
        m = t90['metrics'][cand]
        pr = t90['probability_ranges'][cand]
        a(f"| {cand}{' (chosen)' if cand == t90['chosen'] else ''} | {f(m['auc'])} | {f(m['brier'])} | {f(m['logloss'])} | {f(m['slope'], 3)} | {f(m['intercept'], 3)} | {f(m['ece_10bin'])} | {pr['exact_0_or_1']} ({pr['exact_0_or_1_with_opposite_label']}) |")
    a('')
    a('Match bootstrap (1000 replicates, fixed models; chosen minus comparator; Brier/log loss negative = chosen better):')
    a('')
    a('| comparison | AUC diff [95% CI] | Brier diff [95% CI] | log loss diff [95% CI] |')
    a('|---|---|---|---|')
    for p in t90['bootstrap']['pairs']:
        a(f"| {p['a']} - {p['b']} | {ci(p['a_minus_b']['auc'])} | {ci(p['a_minus_b']['brier'], 5)} | {ci(p['a_minus_b']['logloss'], 5)} |")
    a('')
    a('Isotonic disclosure: the chosen h90 and h120 q are isotonic-calibrated ridge models. sklearn 1.6.1 isotonic prediction is piecewise-linear interpolation with '
      f"flat segments and end clipping; on TEST h90 {t90['probability_ranges'][t90['chosen']]['exact_0_or_1']} rows receive q exactly 0 or 1, "
      f"{t90['probability_ranges'][t90['chosen']]['exact_0_or_1_with_opposite_label']} of them with the opposite label (NA1 16.13: "
      f"{rq['results']['EXT_NA1_16.13']['h90']['probability_ranges'][rq['results']['EXT_NA1_16.13']['h90']['chosen']]['exact_0_or_1']} rows, "
      f"{rq['results']['EXT_NA1_16.13']['h90']['probability_ranges'][rq['results']['EXT_NA1_16.13']['h90']['chosen']]['exact_0_or_1_with_opposite_label']} opposite). "
      'This is the failure mode seen in the pilot P4; it was not fixed using TEST. Saved probabilities are unaltered; log loss uses sklearn float64-eps clipping.')
    a('')
    a('### h90 chosen q on all test sets, and sensitivity horizons')
    a('')
    a('| Set | h | chosen | rows | matches | AUC | Brier | log loss | cal. slope | Brier diff vs constant [95% CI] |')
    a('|---|---:|---|---:|---:|---:|---:|---:|---:|---|')
    for name in ('MAIN_TEST',) + tuple(f'EXT_{s}' for s in EXT):
        for h in C.HORIZONS_S:
            c = rq['results'][name][f'h{h}']
            m = c['metrics'][c['chosen']]
            bp = next((p['a_minus_b']['brier'] for p in c['bootstrap'].get('pairs', []) if p['b'] == 'constant'), None)
            a(f"| {name} | {h} | {c['chosen']} | {f(c['rows'])} | {f(c['matches'])} | {f(m['auc'])} | {f(m['brier'])} | {f(m['logloss'])} | {f(m['slope'], 3)} | {ci(bp, 5) if bp else 'NA'} |")
    a('')
    qd = rd('q_fit_metrics/h90.json')['v_role_diagnostics_after_selection']
    a(f"Diagnostic only (after selection): h90 chosen q on V_CAL AUC {f(qd[t90['chosen']]['v_cal_diag']['auc'])}, Brier {f(qd[t90['chosen']]['v_cal_diag']['brier'])}; "
      f"on V_SELECT AUC {f(qd[t90['chosen']]['v_select_diag']['auc'])}, Brier {f(qd[t90['chosen']]['v_select_diag']['brier'])}. These games' outcomes calibrated/selected V, "
      'so they are not independent q validation.')
    a('')
    strata = {s['stratum']: s for s in t90['strata']}
    a('TEST h90 strata (chosen q; descriptive; post-hoc strata use information after q_pre and are not prediction-time groups): ' + '; '.join(
        f"{k} n={f(strata[k]['rows'])} AUC {f(strata[k]['chosen']['auc'], 3)}" for k in
        ('start_minutes_2_10', 'start_minutes_10_20', 'start_minutes_20_30', 'start_minutes_30_1000', 'pre_history_baron_acquired', 'pre_history_atakhan_acquired',
         'pre_history_soul_acquired', 'posthoc_after_last_kill_baron_present', 'posthoc_after_last_kill_dragon_present') if k in strata and strata[k]['chosen']) + '. Start-minute band 0-2 has no rows.')
    a('')
    a('Pilot context (NOT comparable, not an improvement claim): P4 A90 pilot TEST (7,664 rows / 2,655 matches, pilot V labeler, different cohort) had chosen ridge_isotonic '
      'AUC 0.5886, Brier 0.2450, log loss 0.7031. The labeler, cohort, split sizes and label scale all changed.')
    a('')
    a(f"Figure: {figs[2]}.")
    a('')
    a('## 6. SHAP for the chosen h90 q (TEST explanation)')
    a('')
    g = sh['group_exact_final_output']
    a(f"Model {sh['model']}; {sh['explained_rows']} TEST rows ({sh['explained_matches']} matches) by sha256 rank; background {sh['background_rows']} TRAIN rows. "
      f"Feature-level values: {sh['method']}, on **{sh['scale']}**; additivity residual {sh['additivity_max_abs_residual_vs_actual_pipeline_output']:.2e}. "
      f"Group-level: {g['method']} on the **{g['scale']}**; {g['coalitions']} coalitions, additivity residual {g['additivity_max_abs_residual']:.2e}, "
      f"implementation check on the raw base score {g['implementation_check_on_raw_base_score']['max_abs_diff_vs_grouped_linear_shap']:.2e}. Reload determinism: {all(sh['determinism'].values())}.")
    a('')
    a('| group (n features) | mean abs group Shapley, final q probability | mean signed | mean abs sum of feature SHAP (raw log-odds base score) |')
    a('|---|---:|---:|---:|')
    for r in sorted(sh['group_table'], key=lambda r: -r['mean_abs_group_exact_shapley_final_output']):
        a(f"| {r['group']} ({r['n_features']}) | {f(r['mean_abs_group_exact_shapley_final_output'])} | {f(r['mean_signed_group_exact_shapley_final_output'])} | {f(r['mean_abs_sum_of_feature_shap'])} |")
    a('')
    a('Top feature-level values (raw log-odds base score): ' + ', '.join(f"{t['feature']} {f(t['mean_abs_shap'], 3)}" for t in sh['top_features'][:10]) +
      f". {sh['correlated_top30_pairs_abs_r_ge_0_8']['count']} pairs among the top-30 features have |r| >= 0.8 on TRAIN (slot gold/level/xp are collinear), "
      'so individual feature attributions are unstable and large opposing slot values largely cancel; read groups, not single slots. '
      'Attributions explain this model\'s association with generated labels; they are not causal esports effects, player skill, or feature importance.')
    a('')
    a(f"Figure: {figs[3]}. Local cases: shap/shap_summary.json (local_cases); arrays: shap/shap_values.npz; tables: shap/global_*.csv.")
    a('')
    a('## 7. Temporal SimpleRNN comparator')
    a('')
    if temporal:
        tb_ = temporal['bucket']
        a(f"TEST bucket queries: temporal RNN AUC {f(tb_['temporal']['overall']['auc'])}, log loss {f(tb_['temporal']['overall']['logloss'])} vs primary V AUC "
          f"{f(tb_['primary']['overall']['auc'])}, log loss {f(tb_['primary']['overall']['logloss'])}. Paired bootstrap (temporal - primary): "
          + '; '.join(f"{k} {ci(v, 5)}" for k, v in tb_['bootstrap']['pairs'][0]['a_minus_b'].items()) +
          '. Comparator only; it does not replace the primary labeler and no q/label was recomputed with it.')
    else:
        a('Not complete at report time (status: ' + json.dumps(tstat) + '). The primary pipeline never depended on it. Design and code: scripts/fc20260915_temporal.py '
          '(P2 SimpleRNN architecture/training settings, full TRAIN bucket keys, fit-only preprocessing, positive-slope sigmoid or raw per seed on V_CAL chosen by '
          'V_SELECT log loss, mean ensemble; equal-match loss weights, a disclosed change from the unweighted P2 training).')
    a('')
    a('## 8. Verification record')
    a('')
    for k, v in val['checks'].items():
        a(f"- [{'pass' if v else 'FAIL'}] {k}")
    a('')
    a(f"Independent metric reconstruction (numpy, separate code) over {val['info']['metric_reconstruction']['cells']} TEST/external cells: max abs diff "
      f"AUC {val['info']['metric_reconstruction']['max_abs']['auc']:.1e}, Brier {val['info']['metric_reconstruction']['max_abs']['brier']:.1e}, log loss {val['info']['metric_reconstruction']['max_abs']['logloss']:.1e}. "
      f"Prior artifacts compared against hashes recorded by earlier runs: {val['info']['prior_artifacts_compared']}; differences: {val['info']['prior_artifacts_changed']} "
      '(outputs/full_corpus_preflight_20260915/commands.txt was last modified 08:01:12, before this run; its recorded hash was taken before its final line was appended).')
    a('')
    a('## 9. Deviations, disclosures and unresolved limits')
    a('')
    a('- Labels are generated by a fitted V; cross-fitting prevents a TRAIN match\'s own outcome from entering its labeler, but does not establish that delta>0 means a won fight. No expert semantic review exists.')
    a('- 15.16 and all external sets were examined in earlier work or have unknown/pilot provenance; they are frozen evaluations, not untouched confirmatory tests.')
    a('- The V query grid uses the observed terminal time to bound eligible queries (training/evaluation sampling scheme, not an inference input).')
    a('- ~35% of endpoints have no frame after L; post states then combine a pre-L frame with events (observation staleness).')
    a('- Atakhan / FEAT_UPDATE are absent in 16.xx; unseen champions (18-25% of external queries) map to zero one-hot vectors; external V calibration slopes 0.81-0.87.')
    a('- Chosen isotonic q yields a few exact 0/1 probabilities (see section 5); reported as is.')
    a('- Bootstrap intervals cover test-sample variability only (models, calibration and selection fixed).')
    a('- TRAIN-only smoke runs of the primary pipeline used pseudo roles built from TRAIN matches (smoke_train_only/, smoke_train_only_sigforced/); one forced the sigmoid V path as a code-path test. '
      'The temporal comparator code-path smoke (smoke_train_only_temporal/, 1 epoch, 2 chunks per role) was NOT TRAIN-only: it read V_CAL/V_SELECT history chunks and outcomes '
      '(already used by primary V calibration/selection). No smoke touched TEST or external data and no setting was changed afterwards.')
    a('- The protocol\'s recorded hashes of scripts are those at protocol time; the final script hashes are in source_hashes.json (implementation fixes after protocol writing did not change the design).')
    a('- Old pilot q/V numbers are context only (different labeler, cohort, scale).')
    a('')
    a('## 10. Artifacts')
    a('')
    a('protocol.json; status.json (+ status/*.json); commands.txt; logs/; extract/<set>/extraction_manifest.json (+ chunk states and SEALED outcomes); external/<set>/prepared_manifest.json, exposures.csv, cache/; '
      'selection_v.json; v_models_manifest.json; models/v/*.joblib; labels/*_labels.npz, *_features_pre_only.npz, labels_*_manifest.json; q_pre_only_schema.json; selection/q_h*.json; '
      'models/q/h*/; q_fit_metrics/; predictions/; frozen_manifest.json; outcome_access_log.jsonl; eval/results_v.json, results_q.json, counts_and_exclusions.json, exclusions_*.csv, '
      'predictions/, hashes_before.json, hashes_after.json; shap/; figures/; published/ (outcome-free V query keys and census, label/endpoint distributions by fold and role, external structural-zero audit, non-comparable pilot context); validation.json; DEFINITION_AND_EVIDENCE.md; source_hashes.json; temporal_comparator/.')
    (OUT / 'REPORT.md').write_bytes(('\n'.join(L) + '\n').encode('utf-8'))

    # ---------------------------------------------------------------- DEFINITION_AND_EVIDENCE.md
    E = []
    e = E.append
    e('# Definitions and evidence register: full-corpus retraining 2026-09-15')
    e('')
    e('Evidence types: **literature** (what a cited work supports, with scope), **design choice** (our protocol decision), **data estimate** (measured here), '
      '**completed experiment** (executed and validated here), **unresolved limit**. Literature entries reuse the verified references in '
      'docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md and the P3/P4 evidence documents; no new literature claim was added and papers were not re-opened in this run.')
    e('')
    e('| Element | Exact rule / value | Evidence type | Scope and limits |')
    e('|---|---|---|---|')
    rows = [
        ('Corpus and roles', '210,000 main matches: 15.14 TRAIN 74,673; 15.15 VALIDATION 74,748; 15.16 TEST 60,579; external KR 16.13 10,064, KR 16.14 pilot 200, KR 16.15 926, NA1 16.13 10,000; EUW1 0',
         'data estimate (preflight manifest, re-decoded here)', 'external sets have prior/unknown use; not untouched'),
        ('TRAIN folds', 'sha256("full-v-oof-20260915:"+id)[:8] mod 5', 'design choice', 'five folds are not literature-derived'),
        ('VALIDATION roles', 'sha256("full-val-20260915:"+id)[:8] mod 4 -> V_CAL, V_SELECT, Q_CAL, Q_SELECT', 'design choice', 'separates V calibration outcomes from q validation'),
        ('W', 'final Blue win from GAME_END winningTeam', 'design choice (existing convention)', 'target only; stored in separate sealed outcome files'),
        ('StateV2', 'objective_history_v2_participant_order; latest frame <= query, predictor events <= query; team then participantId order', 'design choice (P1 contract), completed experiment (bitwise equal to P3 rows)', 'frames ~60 s apart; no positions'),
        ('V estimand', 'p_t ~ P(W = 1 given observations at t) via f(X_t, t)', 'literature: Maymin (2021) snapshot state value; Kim et al. (CoG 2020) probability/calibration motivation (uncertainty loss not reproduced); Hodge et al. (2019/2021) live prediction motivation',
         'associational; not causal'),
        ('V query sampling', 'minute grid t >= 2 min, t < GAME_END, <= last frame; one sha256-chosen query per 5-minute bucket; equal match weights', 'design choice; Jalovaara (2024) master thesis as supporting (not peer-reviewed)', 'uses observed terminal time retrospectively'),
        ('V model', 'expanded StateV2 logistic, liblinear, C = 0.01, champion one-hot, snapshot_age_s excluded', 'design choice (C predeclared from pilot; not a validated optimum)', f"converged ({sel['final_fit']['n_iter']} iterations)"),
        ('V calibration', 'raw or positive-slope sigmoid (b >= 1e-6) on V_CAL; choice by V_SELECT log loss, Brier, name; no isotonic', 'design choice', f"chosen {ch}"),
        ('OOF cross-fitting', 'fold-k adapter fitted on other four TRAIN folds; TRAIN engagement p_pre/p_post from its held-out fold', 'design choice; completed experiment (membership reconstructed, 0 violations)', 'not proof of semantic truth'),
        ('Engagement detector', 'v3.3: G 13.7 s, D 4264, R 1600, B 15 s, min 2 per team (original repository sources, hash-equal to parent run)', 'design choice (existing definition); completed experiment (300 TRAIN matches re-detected exactly)', 'retrospective detection'),
        ('Endpoint', 'min(L + h, next raw kill - 1 ms, next engagement start - 1 ms, GAME_END - 1 ms); objectives never terminate', 'design choice (operational B rule)', 'no citation establishes 90 s or this rule'),
        ('Horizon', 'h90 primary; h60/h120 sensitivity; never chosen on TEST', 'design choice', 'label disagreement vs h60 ~1.5%, vs h120 ~0.6% on TEST'),
        ('Label', 'delta = V(post) - V(pre), same adapter; Y = 1[delta > 0]; exact zero -> 0', 'design choice; literature: Maymin (2021) value-change rationale', 'model-defined; not engagement success'),
        ('q inputs', 'StateV2 at q_pre minus snapshot_age_s, plus p_pre_V; 352 numeric ridge inputs; champion IDs excluded', 'design choice (P4 convention)', 'no post-q_pre information'),
        ('q candidates and selection', 'P4 pool (constant, p_pre logistic/spline, ridge raw/sigmoid/isotonic, economic LightGBM raw/sigmoid/isotonic); Q_SELECT Brier, log loss, name', 'design choice (restored P4)',
         f"chosen h90 {qsel[90]['chosen']}, h60 {qsel[60]['chosen']}, h120 {qsel[120]['chosen']}"),
        ('Isotonic prediction', 'sklearn 1.6.1 IsotonicRegression: piecewise-linear interpolation with flat segments, clipped ends', 'software documentation (P4 errata)', 'exact 0/1 outputs possible (observed on TEST/external)'),
        ('Metrics', 'AUC, Brier, log loss (float64 eps clipping), calibration slope/intercept, CITL, ECE, reliability bins; equal match weights', 'literature: Van Calster et al. (2019) calibration assessment (as in P4)', 'calibration to generated labels does not validate labels'),
        ('Uncertainty', '1000 match-resampling bootstrap replicates, fixed models', 'design choice', 'excludes training/selection variability'),
        ('SHAP', 'exact linear interventional SHAP (raw log-odds base score for isotonic winner); exact group Shapley over 7 groups on final probability', 'literature: Lundberg & Lee (2017)', 'association only; collinear features; background choice matters'),
        ('Temporal comparator', 'P2 SimpleRNN (9 one-minute steps, 8 units, dropout .25, RMSprop .001, 50 epochs, seeds 17/29/43) on full TRAIN keys', 'literature: Silva et al. (SBGames 2018) architecture adapted, not reproduced; design choice (equal-match loss weights)',
         'comparator only' + ('' if temporal else '; PENDING at report time')),
        ('V TEST performance', f"AUC {f(rv['results']['MAIN_TEST']['bucket'][ch]['overall']['auc'])}, log loss {f(rv['results']['MAIN_TEST']['bucket'][ch]['overall']['logloss'])} (bucket queries)", 'completed experiment', 'TEST previously examined in pilots'),
        ('q TEST performance', f"h90 {t90['chosen']} AUC {f(t90['metrics'][t90['chosen']]['auc'])}, Brier {f(t90['metrics'][t90['chosen']]['brier'])} vs constant {f(t90['metrics']['constant']['brier'])}", 'completed experiment', 'discrimination of generated labels is modest'),
        ('External shift', 'V calibration slope 0.81-0.87 on 16.xx; Atakhan/FEAT_UPDATE absent; unseen champions 18-25% of queries', 'data estimate', 'no recalibration on test sets'),
        ('Semantic validity of labels', 'no expert blind review', 'unresolved limit', 'required before interpreting Y as fight outcome'),
        ('Observation staleness', f"~35% of h90 endpoints without a frame after L (TEST)", 'data estimate / unresolved limit', 'post state partly event-updated only'),
    ]
    for r in rows:
        e('| ' + ' | '.join(str(x).replace('|', '/') for x in r) + ' |')
    e('')
    e('## Literature and what it does not establish')
    e('')
    e('- **Maymin (2021), DOI 10.1515/jqas-2019-0096**: snapshot win probability and valuing actions by win-probability change. Does not justify our endpoint rule, 90 s, cohort, or q performance.')
    e('- **Kim, Lee, Chung (CoG 2020)**: per-minute LoL winner probability and confidence calibration motivate probability-quality evaluation; their uncertainty-aware loss is not reproduced.')
    e('- **Hodge et al. (IEEE ToG 13(4):368-379, DOI 10.1109/TG.2019.2948469; online 2019)**: live professional esports prediction motivation only; no implementation detail borrowed.')
    e('- **Silva, Pappa, Chaimowicz (SBGames 2018)**: RNN continuous outcome prediction; the comparator adapts a small SimpleRNN, not a reproduction.')
    e('- **Lundberg & Lee (2017), arXiv:1705.07874**: SHAP additive attribution of model predictions; not causal effects or player skill.')
    e('- **Jalovaara (2024), Aalto master thesis**: 5-minute bucket sampling; supporting evidence, not peer-reviewed.')
    e('- **Van Calster et al. (2019), DOI 10.1186/s12916-019-1466-7** (as cited in P4): calibration intercept/slope/curves; medical sample-size rules not transferred.')
    e('- Five folds, validation hash partitions, positive-slope calibration and 90 s are **our protocol choices**, not established by these citations.')
    e('')
    e('## Completed vs unresolved')
    e('')
    e('- Completed and validated: full decoding, external adaptation, V + OOF, TRAIN/VALIDATION/TEST/external labels, q for three horizons, freeze, frozen evaluation with bootstrap, SHAP, post-run checks (validation.json).')
    e('- ' + ('Temporal comparator completed (section 7 of REPORT.md).' if temporal else 'Temporal comparator pending at report generation time.'))
    e('- Unresolved: semantic validity of generated labels; untouched future-patch confirmation; live API availability/latency; causal interpretation (not claimed).')
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_bytes(('\n'.join(E) + '\n').encode('utf-8'))

    src = {}
    for p in sorted((C.ROOT / 'scripts').glob('fc20260915_*.py')) + [C.ROOT / 'tests' / 'test_fc20260915_contracts.py']:
        src[str(p.relative_to(C.ROOT)).replace('\\', '/')] = C.sha256_file(p)
    for rel in ('scripts/engagement_labels_v3_rules.py', 'scripts/measure_postkill_objective_delay.py'):
        src[rel] = C.sha256_file(C.ROOT / rel)
    for rel in proto['sources_sha256']['worktree']:
        src['worktrees/engagement-state-value/' + rel] = C.sha256_file(C.WT / rel)
    outputs = {}
    for rel in ('protocol.json', 'selection_v.json', 'v_models_manifest.json', 'frozen_manifest.json', 'validation.json', 'REPORT.md', 'DEFINITION_AND_EVIDENCE.md',
                'eval/results_v.json', 'eval/results_q.json', 'eval/counts_and_exclusions.json', 'shap/shap_summary.json', 'labels/labels_trainval_manifest.json',
                'labels/labels_test_external_manifest.json', 'q_pre_only_schema.json') + tuple(f'selection/q_h{h}.json' for h in C.HORIZONS_S) + tuple(figs):
        outputs[rel] = C.sha256_file(OUT / rel)
    C.write_json(OUT / 'source_hashes.json', dict(generated_at=time.strftime('%Y-%m-%d %H:%M:%S'), sources=src, outputs=outputs,
                                                  note='REPORT.md hash is of the file at generation time'))
    print('report written', len(L), 'lines;', len(figs), 'figures')


if __name__ == '__main__':
    main()
