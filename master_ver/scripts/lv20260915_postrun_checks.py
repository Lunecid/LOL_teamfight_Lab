"""Label validity stage V: post-run verification -> validation.json.

Independent re-checks of data roles, OOF own-match exclusion, preprocessing membership, same-adapter endpoints,
feature lists, parent immutability, cohort joins and valid masks, zero-delta/tie logic, raw fixture counting,
normalization sources (TRAIN cache + external raw timelines), reload/re-prediction, recomputed metrics and bootstrap,
and reviewer blind-field absence. Failures are recorded, never hidden.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import lv20260915_analysis as A  # noqa: E402
import lv20260915_common as K  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

CHECKS = {}


def check(name, ok, **detail):
    CHECKS[name] = dict(pass_=bool(ok), **detail)
    return ok


def jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def main():
    st = K.Status('postrun_checks')
    K.log_command()
    try:
        fz = C.read_json(K.frozen_manifest_path())
        frozen_at = fz['frozen_at']
        P = K.ParentReadOnly()
        names = P.manifest('MAIN')['names']
        # ------------------------------------------------ 1 data roles / access logs
        st.update('running', 'access_logs')
        acc = jsonl(K.OUT / 'outcome_access_log.jsonl')
        pre = [a for a in acc if a['time'] < frozen_at]
        bad_pre = [a for a in pre if a['set'] != 'MAIN' or not a['sub_roles'] or not set(a['sub_roles']) <= K.ParentReadOnly.PRE_FREEZE_OUTCOME_ROLES]
        post_sealed_before = [a for a in acc if (a['set'] != 'MAIN' or (a['sub_roles'] and not set(a['sub_roles']) <= K.ParentReadOnly.PRE_FREEZE_OUTCOME_ROLES))
                              and not a['this_study_frozen_manifest_exists']]
        lab_acc = jsonl(K.OUT / 'label_access_log.jsonl')
        sealed_before = [a for a in lab_acc if a['set'] in K.SEALED_LABEL_SETS and (a['time'] < frozen_at or not a['this_study_frozen_manifest_exists'])]
        check('data_roles_and_freeze_gate', not bad_pre and not post_sealed_before and not sealed_before,
              outcome_records=len(acc), pre_freeze_records=len(pre), label_records=len(lab_acc), frozen_at=frozen_at,
              violations=(bad_pre + post_sealed_before + sealed_before)[:5])
        # ------------------------------------------------ 2/3 V model membership (full)
        st.update('running', 'v_models')
        from lv20260915_check_v_models import check_models
        vm = check_models(False)
        C.write_json(K.OUT / 'contracts' / 'v_model_checks_postrun.json', vm)
        check('v_models_membership_preprocessing_reload', vm['all_pass'], models={M: dict(pass_=v['pass'], chosen=v['chosen'], converged_all=v['converged_all'])
                                                                                for M, v in vm['models'].items()})
        check('fits_converged', all(v['converged_all'] for v in vm['models'].values()))
        prev = C.read_json(K.OUT / 'contracts' / 'v_model_checks_full.json')
        # timestamps have 1 s resolution (check and freeze ran in the same second); the frozen manifest records the file hash
        check('v_model_checks_before_freeze_passed', prev['all_pass'] and prev['checked_at'] <= frozen_at
              and C.sha256_file(K.OUT / 'contracts' / 'v_model_checks_full.json') == fz['contracts']['v_model_checks_full_sha256'],
              checked_at=prev['checked_at'], frozen_at=frozen_at, hash_recorded_in_frozen_manifest=True)
        # ------------------------------------------------ 2 OOF own-match exclusion on engagement rows
        st.update('running', 'oof_rows')
        man = C.read_json(K.OUT / 'alt_labels' / 'alt_labels_manifest.json')
        oof = {}
        for name in K.LABEL_SETS:
            with np.load(K.OUT / 'alt_labels' / f'{name}_alt_labels.npz', allow_pickle=False) as z:
                ids = {m: z[f'adapter_id_{m}'].astype(str) for m in K.ALL_V}
                shas = {m: z[f'adapter_sha256_{m}'].astype(str) for m in K.ALL_V}
                match = z['match'].astype(str)
            if name == 'MAIN_TRAIN':
                fold = np.asarray([C.train_fold(x) for x in match])
                expect = np.char.add('oof_fold', fold.astype(str))
            else:
                expect = np.full(len(match), 'final')
            r = {m: bool(np.array_equal(ids[m], expect)) for m in K.ALL_V}
            for M in K.MODELS:
                ref = {'final': fz['models'][M]['final_sha256'], **{f'oof_fold{k}': fz['models'][M]['oof_sha256'][f'fold{k}'] for k in range(C.N_FOLDS)}}
                r[f'{M}_sha_matches_frozen'] = bool(np.all(np.asarray([ref[a] for a in ids[M].tolist()]) == shas[M]))
            oof[name] = r
        held = {M: all(a.get('held_out_fold_ok') and a.get('held_out_matches_absent_from_fit') for aid, a in vm['models'][M]['adapters'].items() if aid != 'final')
                for M in K.MODELS}
        check('oof_own_match_exclusion_rows', all(all(v.values()) for v in oof.values()) and all(held.values()), per_set=oof, fold_membership=held)
        # ------------------------------------------------ 5 feature lists
        fl = C.read_json(K.OUT / 'feature_lists.json')
        feat_ok = fl['B_econ']['features'] == K.econ_features(names) and fl['B_reg']['features'] == fl['A_primary']['features'] == K.expanded_features(names)
        check('feature_lists_exact_and_no_forbidden_channels', feat_ok and C.sha256_file(K.OUT / 'feature_lists.json') == fz['feature_lists_sha256'],
              B_econ_n=len(fl['B_econ']['features']), B_reg_n=len(fl['B_reg']['features']))
        # ------------------------------------------------ 6 parent immutability
        st.update('running', 'parent_hashes')
        before = C.read_json(K.OUT / 'integrity' / 'parent_hashes_before.json')['sha256']
        after = {p: C.sha256_file(K.ROOT / p) for p in before}
        changed = [p for p in before if after[p] != before[p]]
        C.write_json(K.OUT / 'integrity' / 'parent_hashes_after.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), files=len(after), sha256=after))
        inv_b = C.read_json(K.OUT / 'integrity' / 'parent_inventory_before.json')['inventory']
        inv_a = K.parent_file_inventory([K.FC, K.CR])
        added = sorted(set(inv_a) - set(inv_b))
        removed = sorted(set(inv_b) - set(inv_a))
        modified = sorted(p for p in set(inv_a) & set(inv_b) if inv_a[p] != inv_b[p])
        C.write_json(K.OUT / 'integrity' / 'parent_inventory_after.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), files=len(inv_a), inventory=inv_a))
        check('parent_artifacts_unchanged', not changed and not added and not removed and not modified, hashed_files=len(before),
              inventory_files=len(inv_a), changed=changed[:10], added=added[:10], removed=removed[:10], modified=modified[:10])
        # ------------------------------------------------ 7 identity / cohort / valid masks
        ident_ok = {n: all(v for v in s['identity'].values() if isinstance(v, bool)) for n, s in man['summaries'].items()}
        a_recomp = {n: s['identity']['A_recomputation'] for n, s in man['summaries'].items()}
        check('exact_keys_endpoints_cohort_joins_valid_masks', all(ident_ok.values()) and set(ident_ok) == set(K.LABEL_SETS), per_set=ident_ok)
        check('primary_A_labels_reproduced', all(all(v for k, v in r.items() if isinstance(v, bool)) or
                                                 (all(r[f'h{h}_Y_equal'] for h in K.HS) and (r['p_pre_max_abs_diff'] or 0) <= 1e-12) for r in a_recomp.values()),
              per_set=a_recomp)
        # ------------------------------------------------ 8 zero delta / ties
        zd = {}
        for name in K.LABEL_SETS:
            with np.load(K.OUT / 'alt_labels' / f'{name}_alt_labels.npz', allow_pickle=False) as z:
                for h in K.HS:
                    v = z[f'valid_h{h}'] == 1
                    for m in K.ALL_V:
                        d, y = z[f'delta_h{h}_{m}'], z[f'Y_h{h}_{m}']
                        zd[f'{name}_h{h}_{m}'] = dict(formula_errors=int(np.sum(y[v] != (d[v] > 0))), invalid_not_minus1=int(np.sum(y[~v] != -1)),
                                                      nonfinite_valid=int(np.sum(~np.isfinite(d[v]))), exact_zero=int(np.sum(d[v] == 0)),
                                                      exact_zero_labelled_1=int(np.sum((d[v] == 0) & (y[v] == 1))))
        check('zero_delta_and_label_logic', all(r['formula_errors'] == 0 and r['invalid_not_minus1'] == 0 and r['nonfinite_valid'] == 0
                                                and r['exact_zero_labelled_1'] == 0 for r in zd.values()),
              exact_zero_total={k: v['exact_zero'] for k, v in zd.items() if v['exact_zero']})
        # ------------------------------------------------ 9 raw fixtures
        fx = C.read_json(K.OUT / 'contracts' / 'train_fixture_checks.json')
        check('event_interval_counting_raw_train_fixtures', fx['all_pass'], checks=fx['checks'], rows=fx['engagement_rows'])
        # ------------------------------------------------ 10 normalization: external raw timelines
        st.update('running', 'external_raw_normalization')
        pm = C.read_json(K.FC / 'external' / 'KR_16.15' / 'prepared_manifest.json')
        with open(C.PREFLIGHT / 'external' / 'external_KR_16.15.csv', encoding='utf-8', newline='') as f:
            pre_rows = {r['match_id']: r for r in csv.DictReader(f)}
        rows = sorted((dict(r, raw_folder=pre_rows[r['match_id']]['raw_folder'], timeline_sha256=pre_rows[r['match_id']]['timeline_sha256'])
                       for r in pm['matches'] if r['cache_status'] == 'written'),
                      key=lambda r: hashlib.sha256(('lv20260915-extfix:' + r['match_id']).encode()).hexdigest())[:5]
        from lv20260915_train_fixtures import DEN
        sys.path.insert(0, str(C.WT))
        os.environ['LOL_CFG_PRESET'] = 'v3.3'
        from core.config import NODE_FEATURE_NAMES
        nn = list(NODE_FEATURE_NAMES)
        ext = dict(matches=len(rows), frames=0, max_abs_diff={k: 0.0 for k in ('totalGold', 'xp', 'currentGold', 'level')}, timestamp_mismatch=0,
                   raw_files_unchanged=True)
        K._append_jsonl(K.OUT / 'outcome_access_log.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), kind='raw_timeline_fixture',
                                                                 set='KR_16.15', sub_roles=['EXTERNAL'], purpose='normalizer trace only (GAME_END not read)',
                                                                 rows=len(rows), this_study_frozen_manifest_exists=True))
        for r in rows:
            raw = Path(r['raw_folder']) / 'timeline' / f"{r['match_id']}.json"
            h0 = C.sha256_file(raw)
            ext['raw_files_unchanged'] &= h0 == r['timeline_sha256']   # also equals the frozen preflight hash
            tl = json.loads(raw.read_text(encoding='utf-8'))
            frames = tl['info']['frames']
            with np.load(K.FC / 'external' / 'KR_16.15' / 'cache' / f"{r['match_id']}.npz", allow_pickle=False) as z:
                ts = z['minute_ts']
                node = z['node_minute'].astype(np.float64)
            ext['frames'] += len(ts)
            for t in range(min(len(ts), len(frames))):
                if int(frames[t]['timestamp']) != int(ts[t]):
                    ext['timestamp_mismatch'] += 1
                    continue
                for pid_s, pf in frames[t]['participantFrames'].items():
                    pid = int(pid_s)
                    for raw_k, node_k, den in (('totalGold', 'totalGold_norm', DEN['totalGold_norm']), ('xp', 'xp_norm', DEN['xp_norm']),
                                               ('currentGold', 'curGold_norm', DEN['curGold_norm']), ('level', 'level_norm', DEN['level_norm'])):
                        d = abs(node[t, pid - 1, nn.index(node_k)] * den - float(pf.get(raw_k, 0)))
                        ext['max_abs_diff'][raw_k] = max(ext['max_abs_diff'][raw_k], float(d))
            ext['raw_files_unchanged'] &= C.sha256_file(raw) == h0
        check('normalization_source_traced', fx['checks']['gold_normalizer_25000_reproduces_gold_team_minute'] and fx['checks']['xp_grid_20000']
              and ext['timestamp_mismatch'] == 0 and all(v <= 0.05 for v in ext['max_abs_diff'].values()) and ext['raw_files_unchanged'],
              train_cache=fx['gold_team_minute'], external_raw=ext, source=fx['normalizer_source'])
        # ------------------------------------------------ 4/11 same-adapter endpoints: re-predict samples; npz reload
        st.update('running', 'repredict')
        paths = {}
        vman = C.read_json(K.FC / 'v_models_manifest.json')
        rep = {}
        for name, set_id, roles in (('MAIN_TRAIN', 'MAIN', [f'fold{k}' for k in range(C.N_FOLDS)]), ('MAIN_TEST', 'MAIN', ['TEST']),
                                    ('EXT_KR_16.15', 'KR_16.15', ['EXTERNAL'])):
            E = P.load_engagements(set_id, roles, states=True, counts=False)
            with np.load(K.OUT / 'alt_labels' / f'{name}_alt_labels.npz', allow_pickle=False) as z:
                Z = {k: z[k] for k in z.files}
            idx = np.flatnonzero(Z['valid_h90'] == 1)[::97]
            rr = {}
            for M in K.MODELS:
                eq = diff = 0
                for aid in sorted(set(Z[f'adapter_id_{M}'][idx].tolist())):
                    sub = idx[Z[f'adapter_id_{M}'][idx] == aid]
                    sha = Z[f'adapter_sha256_{M}'][sub[0]]
                    path = fz['models'][M]['final_path'] if aid == 'final' else fz['models'][M]['oof_paths'][aid.replace('oof_', '')]
                    ad = paths.get((M, aid)) or K.load_alt_adapter(K.OUT / path, str(sha), M)
                    paths[(M, aid)] = ad
                    pp = ad.predict_matrix(E['X_pre'][sub], names, C.STATE_VERSION)
                    pq = ad.predict_matrix(E['X_post_h90'][sub], names, C.STATE_VERSION)
                    eq += int(np.sum((pp == Z[f'p_pre_{M}'][sub]) & (pq == Z[f'p_post_h90_{M}'][sub])))
                    diff = max(diff, float(np.max(np.abs(pp - Z[f'p_pre_{M}'][sub]))), float(np.max(np.abs(pq - Z[f'p_post_h90_{M}'][sub]))))
                    if not np.array_equal((pq - pp) > 0, Z[f'Y_h90_{M}'][sub] == 1):
                        diff = max(diff, 1.0)
                rr[M] = dict(rows=int(len(idx)), bitwise_equal_rows=eq, max_abs_diff=diff)
            rep[name] = rr
            E = None
        check('same_adapter_endpoints_repredicted', all(v['max_abs_diff'] <= 1e-12 for r in rep.values() for v in r.values()), samples=rep)
        reload_ok = {}
        for name, s in man['summaries'].items():
            reload_ok[name] = C.sha256_file(K.OUT / 'alt_labels' / f'{name}_alt_labels.npz') == s['alt_labels_sha256'] and \
                C.sha256_file(K.OUT / 'observed' / f'{name}_observed.npz') == s['observed_sha256']
        check('saved_predictions_reload_hashes', all(reload_ok.values()), per_set=reload_ok)
        # ------------------------------------------------ 12 recomputed metrics / bootstrap
        st.update('running', 'recompute_metrics')
        ve = C.read_json(K.OUT / 'results' / 'v_eval.json')
        rec = {}
        with np.load(K.OUT / 'eval' / 'v_predictions' / 'v_MAIN_TEST.npz', allow_pickle=False) as z:
            y = z['winner_blue'].astype(float)
            w = C.weights(z['match'])
            for m in K.ALL_V:
                p = z[f'p_{m}']
                ref = ve['sets']['MAIN_TEST']['by_model'][m]['overall']
                rec[f'V_{m}'] = max(abs(C.auc_direct(y, p, w) - ref['auc']), abs(C.brier_direct(y, p, w) - ref['brier']),
                                    abs(C.logloss_direct(y, p, w) - ref['logloss']))
        with np.load(K.FC / 'eval' / 'predictions' / 'v_MAIN_TEST.npz', allow_pickle=False) as z:
            b = z['is_bucket_sample'] == 1
            rv = C.read_json(K.FC / 'eval' / 'results_v.json')['results']['MAIN_TEST']['bucket']['raw']['overall']
            rec['V_A_equals_parent_results_v'] = max(abs(ve['sets']['MAIN_TEST']['by_model']['A']['overall'][k] - rv[k]) for k in ('auc', 'brier', 'logloss'))
        per = C.read_json(K.OUT / 'results' / 'per_set' / 'MAIN_TEST.json')
        with np.load(K.OUT / 'alt_labels' / 'MAIN_TEST_alt_labels.npz', allow_pickle=False) as z:
            m90 = (z['valid_h90'] == 1) & (z['cohort'] == 1)
            g = z['match'][m90].astype(str)
            dis = (z['Y_h90_A'][m90] != z['Y_h90_B_econ'][m90]).astype(float)
        u, inv = np.unique(g, return_inverse=True)
        per_match = np.bincount(inv, weights=dis) / np.bincount(inv)
        cell = per['agreement']['h90']['T']['A_vs_B_econ']
        rec['agreement_T_B_econ_row'] = abs(dis.mean() - cell['disagreement_row'])
        rec['agreement_T_B_econ_match'] = abs(per_match.mean() - cell['disagreement_match_weighted'])
        naive = A.naive_bootstrap_replicates(g, dis, 5, K.BOOT_SEED)
        draws = per['bootstrap_h90']['T']['_draws_first5']['A_vs_B_econ']
        rec['bootstrap_first5_row'] = max(abs(naive[r][0] - draws['row'][r]) for r in range(5))
        rec['bootstrap_first5_match'] = max(abs(naive[r][1] - draws['match_weighted'][r]) for r in range(5))
        qd = C.read_json(K.OUT / 'results' / 'q_label_dependence.json')
        rec['q_primary_reproduction_all'] = all(v['reproduced'] for c in qd['sets'].values() for v in c['reproduction'].values())
        rec['v_primary_identity_all'] = all(s['primary_identity']['A_max_abs_diff_vs_parent'] <= 1e-12 for k, s in ve['sets'].items() if 'primary_identity' in s)
        check('independent_metric_and_bootstrap_recomputation',
              all(v <= 1e-9 for k, v in rec.items() if isinstance(v, float)) and rec['q_primary_reproduction_all'] and rec['v_primary_identity_all'],
              detail=rec)
        # ------------------------------------------------ 13 reviewer blind absence
        md = (K.OUT / 'review_packet' / 'REVIEWER_PACKET_KO.md').read_text(encoding='utf-8')
        pkey = C.read_json(K.OUT / 'PRIVATE_review_case_key.json')['cases']
        ids_found = [c for c, v in pkey.items() if v['match'] in md or v['match'].split('_')[-1] in md]
        forbidden = [t for t in A.REVIEWER_FORBIDDEN_TOKENS if t in md]
        decimals = re.findall(r'(?<![\d:.])0\.\d{2,}', md)
        with open(K.OUT / 'PRIVATE_review_analysis_key.csv', encoding='utf-8') as f:
            ak = list(csv.DictReader(f))
        leaked_values = [r['case_id'] for r in ak if any(f"{float(r[k]):.3f}" in md for k in ('p_pre_A', 'p_post_h90_A') if float(r[k]) >= 0.1)]
        with open(K.OUT / 'review_packet' / 'review_form_blank.csv', encoding='utf-8') as f:
            form = list(csv.DictReader(f))
        rs = C.read_json(K.OUT / 'review_packet' / 'review_status.json')
        sel = C.read_json(K.OUT / 'results' / 'review_packet_selection.json')
        check('reviewer_blind_fields_absent', not ids_found and not forbidden and not decimals and not leaked_values
              and all(all(v == '' for k, v in r.items() if k != 'case_id') for r in form) and rs['human_review'] == 'UNPERFORMED'
              and rs['ratings_filled'] == 0 and sel['unique_matches'] and sel['raw_files_unchanged'] and len(pkey) == len(form) <= 120,
              cases=len(pkey), shortages=sel['shortages'], ids_found=ids_found[:5], forbidden=forbidden, decimals=decimals[:5],
              leaked_probability_strings=leaked_values[:5])
        # ------------------------------------------------ status
        stat = C.read_json(K.OUT / 'status.json')
        failed = {g: s for g, s in stat['groups'].items() if s.get('state') == 'failed' and g != 'postrun_checks'}
        check('all_stage_groups_complete', not failed and all(s.get('state') == 'complete' for g, s in stat['groups'].items() if g != 'postrun_checks'),
              failed=failed, groups={g: s.get('state') for g, s in stat['groups'].items()})
        n_pass = sum(v['pass_'] for v in CHECKS.values())
        out = dict(role=K.ROLE_TAG, version=K.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), checks=CHECKS, passed=n_pass,
                   total=len(CHECKS), all_pass=n_pass == len(CHECKS),
                   scope='verification of computation, identity and blinding contracts; numeric agreement does not establish construct validity')
        C.write_json(K.OUT / 'validation.json', out)
        st.update('complete' if out['all_pass'] else 'failed', 'postrun_checks', passed=n_pass, total=len(CHECKS),
                  error=None if out['all_pass'] else [k for k, v in CHECKS.items() if not v['pass_']], next_step='report')
        return 0 if out['all_pass'] else 2
    except SystemExit as exc:
        st.update('failed', 'postrun_checks', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'postrun_checks', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
