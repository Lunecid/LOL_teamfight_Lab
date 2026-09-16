"""Full-corpus post-run verification (separate code path from the runners); writes validation.json.

Re-derives from saved artifacts: counts and partition disjointness, selection-before-TEST ordering (file mtimes and
outcome access log), frozen hashes, independent metric reconstruction from saved predictions (numpy only), label
formula identity, prior-artifact immutability against hashes recorded by earlier runs, protocol/input hashes, SHAP
additivity and contract-test evidence. Nothing is refitted.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402

OUT = C.OUT
EXT = ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13')


def rd(p):
    return C.read_json(OUT / p)


def mtime(p):
    return os.path.getmtime(OUT / p)


def main():
    checks, info = {}, {}
    proto = rd('protocol.json')
    # ---- protocol inputs unchanged since the protocol was written
    changed_inputs = {k: v for k, v in proto['inputs_sha256'].items() if v and C.sha256_file(C.ROOT / k) != v}
    changed_gov = {k: v for k, v in proto['governing_sha256'].items() if v and k != 'outputs/full_corpus_preflight_20260915/status.json'
                   and C.sha256_file(C.ROOT / k) != v}
    checks['protocol_inputs_unchanged'] = not changed_inputs
    checks['governing_docs_unchanged_since_protocol'] = not changed_gov
    info['governing_changed'] = list(changed_gov)
    wt_changed = {k: v for k, v in proto['sources_sha256']['worktree'].items() if v and C.sha256_file(C.WT / k) != v}
    checks['worktree_sources_unchanged'] = not wt_changed
    # ---- prior artifacts unchanged vs hashes recorded by earlier runs
    prior = {}
    for rel, sha in C.read_json(C.ROOT / 'outputs/q_v3_baselines/hashes_after.json').items():
        p = C.ROOT / rel
        prior[rel] = p.exists() and C.sha256_file(p) == sha
    p1 = C.read_json(C.ROOT / 'outputs/independent_v2_participant_order/hashes_after.json')
    bases = {'legacy_sources': C.WT, 'old_v_artifacts': C.WT / 'outputs' / 'temporal_winprob_v3_buckets',
             'state_value_v2_fix': C.ROOT / 'outputs' / 'state_value_v2_fix'}
    for grp, files in p1.items():
        for rel, sha in files.items():
            p = (C.ROOT / rel) if rel.startswith('tests/') else bases[grp] / rel
            prior[f'p1_recorded/{grp}/{rel}'] = p.exists() and C.sha256_file(p) == sha
    status_pre = C.read_json(C.ROOT / 'outputs/full_corpus_preflight_20260915/status.json')
    for rel, sha in status_pre['output_sha256'].items():
        p = C.ROOT / 'outputs/full_corpus_preflight_20260915' / rel
        prior['preflight/' + rel] = p.exists() and C.sha256_file(p).lower() == sha.lower()
    run_start = os.path.getmtime(OUT / 'protocol.json')

    def path_of(k):
        if k.startswith('preflight/'):
            return C.ROOT / 'outputs/full_corpus_preflight_20260915' / k[len('preflight/'):]
        if k.startswith('p1_recorded/'):
            _, grp, rel = k.split('/', 2)
            return (C.ROOT / rel) if rel.startswith('tests/') else bases[grp] / rel
        return C.ROOT / k
    diff = {k: dict(exists=path_of(k).exists(), mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path_of(k)))) if path_of(k).exists() else None,
                    modified_after_this_run_started=path_of(k).exists() and os.path.getmtime(path_of(k)) >= run_start)
            for k, v in prior.items() if not v}
    checks['prior_artifacts_not_modified_during_this_run'] = bool(prior) and not any(d['modified_after_this_run_started'] or not d['exists'] for d in diff.values())
    info['prior_artifacts_compared'] = len(prior)
    info['prior_artifacts_changed'] = list(diff)
    info['prior_artifact_differences_vs_recorded'] = dict(diff, note='hash differs from the value recorded by an earlier run; files last modified '
                                                               'before this run started (protocol.json mtime) were not changed by this run')
    run_repo = rd('external/repo_source_check.json')
    checks['original_repo_detector_sources_unchanged'] = run_repo['unchanged'] and all(v for k, v in run_repo['before'].items() if k.endswith('.py') and not k.startswith('scripts/'))
    # ---- extraction completeness
    mm = rd('extract/MAIN/extraction_manifest.json')
    checks['main_all_210000_decoded'] = mm['totals']['matches'] == 210000 and mm['totals']['loaded'] == 210000
    checks['main_exposure_rows_566452'] = mm['totals']['e_rows'] == 566452
    info['main_totals'] = mm['totals']
    ext_ok = {}
    for s in EXT:
        pm = rd(f'external/{s}/prepared_manifest.json')
        em = rd(f'extract/{s}/extraction_manifest.json')
        ext_ok[s] = pm['written'] == pm['expected'] and em['totals']['matches'] == pm['written'] and em['totals']['loaded'] == pm['written']
    checks['external_sets_all_manifest_matches_adapted_and_decoded'] = all(ext_ok.values())
    checks['euw1_reported_unavailable'] = 'blocker' in rd('external/EUW1_complete/prepared_manifest.json')
    checks['detector_train_reproduction_exact'] = rd('external/detector_reproduction_train_check.json')['mismatching_matches'] == 0
    p3 = rd('probes/p3_compare_result.json')
    checks['extraction_bitwise_equal_to_p3_pilot_rows'] = p3['mismatches'] == 0 and p3['compared_row_horizons'] > 0
    # ---- partitions
    counts = rd('eval/counts_and_exclusions.json')['counts']['MAIN']
    folds = {k: v['raw_matches'] for k, v in counts.items() if k.startswith('fold')}
    vals = {k: counts[k]['raw_matches'] for k in C.VAL_ROLES}
    checks['partition_counts_sum_to_manifest'] = sum(folds.values()) == 74673 and sum(vals.values()) == 74748 and counts['TEST']['raw_matches'] == 60579
    info['train_folds'] = folds
    info['validation_roles'] = vals
    ids = {}
    with open(C.MAIN_MANIFEST, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ids.setdefault(r['role'], set()).add(r['match_id'])
    fold_sets = {k: {m for m in ids['TRAIN'] if C.train_fold(m) == k} for k in range(5)}
    role_sets = {r: {m for m in ids['VALIDATION'] if C.validation_role(m) == r} for r in C.VAL_ROLES}
    allsets = [ids['TEST']] + list(fold_sets.values()) + list(role_sets.values())
    checks['partitions_mutually_disjoint'] = sum(len(s) for s in allsets) == len(set().union(*allsets)) == 210000
    # ---- V selection / OOF
    sel = rd('selection_v.json')
    vman = rd('v_models_manifest.json')
    checks['v_final_converged'] = sel['final_fit']['converged'] and not sel['final_fit']['convergence_warnings']
    checks['v_oof_converged_and_membership'] = all(f['fit']['converged'] and f['membership']['overlap'] == 0 for f in vman['folds'].values())
    checks['v_selection_rule_reproduced'] = min(sel['select_scores'], key=lambda k: (sel['select_scores'][k]['logloss'], sel['select_scores'][k]['brier'], k)) == sel['chosen']
    lt = rd('labels/labels_trainval_manifest.json')
    tr = lt['summaries']['MAIN_TRAIN']
    checks['oof_self_outcome_exclusion'] = tr['oof_self_outcome_exclusion']['rows_whose_adapter_fit_includes_match'] == 0 and all(
        v['reconstructed_fit_set_sha_equals_adapter'] for v in tr['oof_self_outcome_exclusion']['adapter_membership'].values())
    # ---- ordering: selections and freeze before TEST/external labels and outcome access
    fz = rd('frozen_manifest.json')
    frozen_t = mtime('frozen_manifest.json')
    test_label_files = ['labels/MAIN_TEST_labels.npz'] + [f'labels/EXT_{s}_labels.npz' for s in EXT]
    checks['freeze_before_test_external_labels'] = all(mtime(p) > frozen_t for p in test_label_files) and not fz['test_labels_existed_before_freeze']
    checks['selections_before_freeze'] = all(mtime(f'selection/q_h{h}.json') <= frozen_t for h in C.HORIZONS_S) and mtime('selection_v.json') < mtime('labels/MAIN_TRAIN_labels.npz')
    access = [json.loads(line) for line in open(OUT / 'outcome_access_log.jsonl', encoding='utf-8')]
    sealed = [a for a in access if a['set'] != 'MAIN' or (a['sub_roles'] is None) or 'TEST' in (a['sub_roles'] or [])]
    checks['sealed_outcomes_only_after_freeze'] = all(a['frozen_manifest_exists'] for a in sealed)
    early_q = [a for a in access if not a['frozen_manifest_exists'] and a['set'] == 'MAIN' and any(r in ('Q_CAL', 'Q_SELECT') for r in (a['sub_roles'] or []))]
    checks['q_role_outcomes_not_used_before_freeze'] = not early_q
    info['outcome_access_log'] = access
    # ---- frozen hashes
    hb, ha = rd('eval/hashes_before.json'), rd('eval/hashes_after.json')
    checks['frozen_model_hashes_unchanged_through_evaluation'] = all(v['equal'] for v in hb.values()) and all(v['equal'] for v in ha.values())
    # ---- independent metric reconstruction from saved predictions
    rq = rd('eval/results_q.json')['results']
    rv = rd('eval/results_v.json')
    recon = {'max_abs': {'auc': 0., 'brier': 0., 'logloss': 0.}, 'cells': 0}

    def upd(y, p, g, ref):
        w = C.weights(g)
        for k, fn in (('auc', C.auc_direct), ('brier', C.brier_direct), ('logloss', C.logloss_direct)):
            v = fn(y, p, w)
            if (v is None) != (ref[k] is None):
                recon['max_abs'][k] = float('inf')
            elif v is not None:
                recon['max_abs'][k] = max(recon['max_abs'][k], abs(v - ref[k]))
        recon['cells'] += 1
    for name in ('MAIN_TEST',) + tuple(f'EXT_{s}' for s in EXT):
        for h in C.HORIZONS_S:
            with np.load(OUT / 'eval' / 'predictions' / f'q_{name}_h{h}.npz', allow_pickle=False) as z:
                y, g = z['y'].astype(float), z['match']
                with np.load(OUT / 'labels' / f'{name}_labels.npz', allow_pickle=False) as lab:
                    v = lab[f'valid_h{h}'] == 1
                    delta = lab[f'delta_h{h}'][v]
                    if not np.array_equal((delta > 0).astype(float), y):
                        recon['max_abs']['label_formula'] = float('inf')
                for cand in C.Q_CANDIDATES:
                    upd(y, z[cand], g, rq[name][f'h{h}']['metrics'][cand])
        vfile = OUT / 'eval' / 'predictions' / f'v_{name}.npz'
        with np.load(vfile, allow_pickle=False) as z:
            b = z['is_bucket_sample'] == 1
            ch = str(z['chosen'])
            upd(z['winner_blue'][b].astype(float), z[f'p_{ch}'][b], z['match'][b], rv['results'][name]['bucket'][ch]['overall'])
    checks['independent_metric_reconstruction_lt_1e-10'] = all(v < 1e-10 for v in recon['max_abs'].values())
    info['metric_reconstruction'] = recon
    # ---- serialization, q predictors, SHAP
    checks['q_serialization_identity_trainval'] = all(all(v.values()) for v in rq['serialization_identity_trainval'].values())
    checks['q_pre_freeze_serialization_and_fit_origins'] = all(all(d.values()) for d in fz['pre_freeze_checks']['serialization_identity'].values()) and \
        all(all(d.values()) for d in fz['pre_freeze_checks']['fitting_origins'].values())
    schema = rd('q_pre_only_schema.json')
    checks['q_no_forbidden_or_categorical_predictors'] = schema['ridge_count'] == 352 and not any(
        any(t in n for t in C.Q_FORBIDDEN_TOKENS) for n in schema['predictor_sets']['ridge'] + schema['predictor_sets']['economic'] if n != C.P_PRE)
    for nm in ('MAIN_TRAIN', 'MAIN_VALIDATION'):
        s_ = lt['summaries'][nm]
        checks[f'labels_{nm}_formula_and_boundaries'] = all(s_['checks'][f'h{h}']['label_formula_exact'] and s_['checks'][f'h{h}']['finite_valid'] for h in C.HORIZONS_S) \
            and s_['checks']['endpoints_monotone_60_90_120'] and not any(s_['checks']['no_raw_kill_in_L_endpoint'].values()) \
            and s_['checks']['pre_snapshot_not_after_query'] and all(s_['checks']['post_snapshot_not_after_endpoint'].values())
    lte = rd('labels/labels_test_external_manifest.json')
    checks['labels_test_external_formula_and_boundaries'] = all(
        all(s_['checks'][f'h{h}']['label_formula_exact'] for h in C.HORIZONS_S) and not any(s_['checks']['no_raw_kill_in_L_endpoint'].values())
        for s_ in lte['summaries'].values() if 'checks' in s_)
    if (OUT / 'shap' / 'shap_summary.json').exists():
        sh = rd('shap/shap_summary.json')
        checks['shap_checks'] = all(sh['checks'].values())
    else:
        checks['shap_checks'] = False
    tests_log = (OUT / 'logs' / 'contract_tests_run1.txt').read_text(encoding='utf-8', errors='replace')
    checks['contract_tests_passed_before_full_fit'] = '32 passed' in tests_log and 'failed' not in tests_log.lower()
    tmp = OUT / 'temporal_comparator' / 'results_temporal.json'
    info['temporal_comparator'] = 'complete' if tmp.exists() else 'pending_or_not_run'
    if tmp.exists():
        hm = rd('temporal_comparator/history_manifest.json')
        ts_ = rd('temporal_comparator/selection_temporal.json')
        checks['temporal_history_query_states_equal_primary_v_states'] = hm['query_rows_checked'] == hm['query_state_equal_float32'] == 986532
        checks['temporal_models_unchanged_after_selection'] = all(C.sha256_file(OUT / 'temporal_comparator' / 'models' / k) == v for k, v in ts_['model_sha256'].items())
        checks['temporal_selection_before_test_evaluation'] = mtime('temporal_comparator/selection_temporal.json') < mtime('temporal_comparator/results_temporal.json')
    failed = [k for k, v in checks.items() if not v]
    val = dict(role=C.ROLE_TAG, version=C.VERSION, generated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
               status='pass' if not failed else 'fail', failed_checks=failed, checks=checks, info=info,
               scope='post-run verification of saved artifacts; the temporal comparator is reported separately and is not a primary check')
    C.write_json(OUT / 'validation.json', val)
    print(json.dumps(dict(status=val['status'], failed=failed, n_checks=len(checks), recon=recon, prior_compared=info['prior_artifacts_compared'],
                          prior_changed=info['prior_artifacts_changed']), indent=1))


if __name__ == '__main__':
    main()
