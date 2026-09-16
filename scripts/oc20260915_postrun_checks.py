"""Objective-channel ablation stage V: post-run verification -> validation.json (failures recorded, never hidden)."""
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
import lv20260915_common as LV  # noqa: E402
import oc20260915_common as K  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime')

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

CHECKS = {}


def check(name, ok, **detail):
    CHECKS[name] = dict(pass_=bool(ok), **detail)


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()] if Path(p).exists() else []


def main():
    st = K.Status('postrun_checks')
    K.log_command()
    try:
        fz = C.read_json(K.frozen_manifest_path())
        frozen_at = fz['frozen_at']
        P = K.parent_reader()
        names = P.manifest('MAIN')['names']
        # 1 access / freeze
        st.update('running', 'access')
        acc = jsonl(K.OUT / 'outcome_access_log.jsonl')
        allowed = LV.ParentReadOnly.PRE_FREEZE_OUTCOME_ROLES
        sealed_rec = [a for a in acc if a['set'] != 'MAIN' or not a['sub_roles'] or not set(a['sub_roles']) <= allowed]
        viol = [a for a in sealed_rec if not a['this_study_frozen_manifest_exists'] or a['time'] < frozen_at]
        lab_acc = jsonl(K.OUT / 'label_access_log.jsonl')
        viol += [a for a in lab_acc if a['set'] in K.SEALED_LABEL_SETS and (a['time'] < frozen_at or not a['this_study_frozen_manifest_exists'])]
        sel = C.read_json(K.OUT / 'selection' / 'selection_secondary_calibration.json')
        man = C.read_json(K.OUT / 'v_models_manifest_B_noobj.json')
        check('data_roles_freeze_gate_and_order', not viol and sel['written_at'] <= man['completed_at'] <= frozen_at,
              outcome_records=len(acc), sealed_outcome_records=len(sealed_rec), label_records=len(lab_acc), frozen_at=frozen_at,
              secondary_selection_at=sel['written_at'], manifest_completed_at=man['completed_at'], violations=viol[:5])
        # 2 models
        st.update('running', 'models')
        from oc20260915_check_v_models import check_models
        vm = check_models(False)
        C.write_json(K.OUT / 'contracts' / 'v_model_checks_postrun.json', vm)
        check('adapters_membership_preprocessing_retained_values_reload', vm['all_pass'], adapters_passed=sum(a['pass'] for a in vm['adapters'].values()),
              adapters=len(vm['adapters']), retained_values_identical_to_A_inputs=vm['retained_values_identical_to_A_inputs'])
        check('frozen_hashes_and_pre_freeze_checks', all(C.sha256_file(K.OUT / p) == s for p, s in fz['adapter_file_sha256'].items())
              and C.sha256_file(K.OUT / 'contracts' / 'v_model_checks_full.json') == fz['contracts']['v_model_checks_full_sha256']
              and C.read_json(K.OUT / 'contracts' / 'v_model_checks_full.json')['all_pass'], adapter_files=len(fz['adapter_file_sha256']))
        check('fits_converged_single_attempt', man['converged_all'] and len(man['final_fit']['attempts']) == 1
              and all(len(v['fit']['attempts']) == 1 for v in man['folds'].values()), final_n_iter=man['final_fit']['n_iter'],
              fold_n_iter=[v['fit']['n_iter'] for v in man['folds'].values()])
        ev = C.read_json(K.OUT / 'feature_evidence.json')
        r, d = K.feature_split(C.read_json(K.FC / 'v_models_manifest.json')['feature_names'])
        check('feature_removal_exact', r == ev['retained_ordered'] == man['feature_names'] and d == [x['name'] for x in ev['dropped_ordered']]
              and (len(r), len(d)) == (185, 176) and C.sha256_file(K.OUT / 'feature_evidence.json') == fz['feature_evidence_sha256'],
              retained=len(r), dropped=len(d))
        # 3 OOF rows
        lm = C.read_json(K.OUT / 'labels' / 'labels_manifest.json')
        oof = {}
        for name in K.LABEL_SETS:
            with np.load(K.OUT / 'labels' / f'{name}_B_noobj_labels.npz', allow_pickle=False) as z:
                ids, shas, match = z['adapter_id_B'].astype(str), z['adapter_sha256_B'].astype(str), z['match'].astype(str)
                ids_s = z['adapter_id_Bsig'].astype(str)
            expect = np.char.add('oof_fold', np.asarray([C.train_fold(m) for m in match]).astype(str)) if name == 'MAIN_TRAIN' else np.full(len(match), 'final')
            ref = {'final': fz['primary']['final_sha256'], **{f'oof_fold{k}': fz['primary']['oof_sha256'][f'fold{k}'] for k in range(C.N_FOLDS)}}
            oof[name] = bool(np.array_equal(ids, expect) and np.array_equal(ids_s, expect) and np.all(np.asarray([ref[a] for a in ids.tolist()]) == shas))
        check('oof_own_match_exclusion_and_adapter_roles', all(oof.values()) and all(a['held_out_ok'] for k, a in vm['adapters'].items()), per_set=oof)
        # 4 parents
        st.update('running', 'parents')
        before = C.read_json(K.OUT / 'integrity' / 'parent_hashes_before.json')['sha256']
        after = {p: C.sha256_file(K.ROOT / p) for p in before}
        changed = [p for p in before if after[p] != before[p]]
        C.write_json(K.OUT / 'integrity' / 'parent_hashes_after.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), sha256=after))
        check('parent_read_targets_unchanged', not changed, hashed=len(before), changed=changed[:10])
        ib = C.read_json(K.OUT / 'integrity' / 'parent_inventory_before.json')['inventory']
        ia = LV.parent_file_inventory(K.parent_roots())
        diffs = sorted(set(ia) ^ set(ib)) + sorted(p for p in set(ia) & set(ib) if ia[p] != ib[p])
        C.write_json(K.OUT / 'integrity' / 'parent_inventory_after.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), files=len(ia), inventory=ia))
        check('parent_inventory_unchanged', not diffs, files=len(ia), differences=diffs[:20])
        # 5 identity, labels, zeros, calibration
        ident = {n: all(v for v in s['identity'].values() if isinstance(v, bool)) and all(s['identity']['A_recomputation'].values()) for n, s in lm['summaries'].items()}
        check('exact_keys_endpoints_masks_and_A_bit_for_bit', all(ident.values()) and set(ident) == set(K.LABEL_SETS), per_set=ident)
        zl = {}
        for name in K.LABEL_SETS:
            with np.load(K.OUT / 'labels' / f'{name}_B_noobj_labels.npz', allow_pickle=False) as z:
                for h in K.HS:
                    v = z[f'valid_h{h}'] == 1
                    for m in ('A', 'B', 'Bsig'):
                        dd, yy = z[f'delta_h{h}_{m}'], z[f'Y_h{h}_{m}']
                        zl[f'{name}/h{h}/{m}'] = dict(formula=int(np.sum(yy[v] != (dd[v] > 0))), invalid=int(np.sum(yy[~v] != -1)),
                                                      nonfinite=int(np.sum(~np.isfinite(dd[v]))), zero=int(np.sum(dd[v] == 0)))
        check('label_formula_invalid_rows_and_exact_zeros', all(x['formula'] == 0 and x['invalid'] == 0 and x['nonfinite'] == 0 for x in zl.values()),
              exact_zero_total=sum(x['zero'] for x in zl.values()))
        inv = {f'{n}/h{h}': s[f'h{h}']['sign_invariance_sigmoid_vs_raw'] for n, s in lm['summaries'].items() for h in K.HS}
        check('shared_monotone_calibration_sign_invariance', all(v['direction_reversal_rows'] == 0 and v['label_mismatch_rows'] == v['collapsed_to_zero_rows'] for v in inv.values()),
              label_mismatch_total=sum(v['label_mismatch_rows'] for v in inv.values()), note='implementation check, not independent label evidence')
        keyid = {n: all(C.read_json(K.OUT / 'results' / 'per_set' / f'{n}.json')['key_identity'].values()) for n in K.LABEL_SETS}
        check('observed_arrays_exact_key_identity', all(keyid.values()), per_set=keyid)
        # 6 re-predict samples
        st.update('running', 'repredict')
        rep = {}
        for name, set_id, roles in (('MAIN_TRAIN', 'MAIN', [f'fold{k}' for k in range(C.N_FOLDS)]), ('MAIN_TEST', 'MAIN', ['TEST']), ('EXT_KR_16.15', 'KR_16.15', ['EXTERNAL'])):
            E = P.load_engagements(set_id, roles, states=True, counts=False)
            with np.load(K.OUT / 'labels' / f'{name}_B_noobj_labels.npz', allow_pickle=False) as z:
                Z = {k: z[k] for k in ('adapter_id_B', 'adapter_sha256_B', 'valid_h90', 'p_pre_B', 'p_post_h90_B', 'Y_h90_B', 'match', 's')}
            if not (np.array_equal(E['match'], Z['match']) and np.array_equal(E['s'], Z['s'])):
                raise SystemExit('re-prediction keys differ')
            idx = np.flatnonzero(Z['valid_h90'] == 1)[::89]
            mx, eq = 0.0, 0
            for aid in sorted(set(Z['adapter_id_B'][idx].tolist())):
                sub = idx[Z['adapter_id_B'][idx] == aid]
                path = fz['primary']['final_path'] if aid == 'final' else fz['primary']['oof_paths'][aid.replace('oof_', '')]
                ad = K.load_b_adapter(K.OUT / path, str(Z['adapter_sha256_B'][sub[0]]))
                pp, pq = ad.predict_matrix(E['X_pre'][sub], names, C.STATE_VERSION), ad.predict_matrix(E['X_post_h90'][sub], names, C.STATE_VERSION)
                eq += int(np.sum((pp == Z['p_pre_B'][sub]) & (pq == Z['p_post_h90_B'][sub])))
                mx = max(mx, float(np.max(np.abs(pp - Z['p_pre_B'][sub]))), float(np.max(np.abs(pq - Z['p_post_h90_B'][sub]))))
                if not np.array_equal((pq - pp) > 0, Z['Y_h90_B'][sub] == 1):
                    mx = max(mx, 1.0)
            rep[name] = dict(rows=int(len(idx)), bitwise_equal=eq, max_abs_diff=mx)
            E = None
        check('same_adapter_endpoints_repredicted', all(v['max_abs_diff'] <= 1e-12 for v in rep.values()), samples=rep)
        check('saved_label_files_reload_hashes', all(C.sha256_file(K.OUT / 'labels' / f'{n}_B_noobj_labels.npz') == s['alt_labels_sha256'] for n, s in lm['summaries'].items()))
        # 7 independent metric / bootstrap recomputation
        st.update('running', 'recompute')
        w = C.read_json(K.OUT / 'results' / 'w_eval.json')
        rec = {}
        with np.load(K.OUT / 'eval' / 'w_predictions' / 'w_MAIN_TEST.npz', allow_pickle=False) as z:
            y, g = z['winner_blue'].astype(float), z['match']
            pA, pB = z['p_A_raw'], z['p_B_noobj_raw']
        wm = C.weights(g)
        for tag, p in (('A_raw', pA), ('B_noobj_raw', pB)):
            for wn, ww in (('match_weighted', wm), ('row_weighted', np.ones(len(y)))):
                ref = w['sets']['MAIN_TEST'][tag][wn]
                rec[f'W_{tag}_{wn}'] = max(abs(C.auc_direct(y, p, ww) - ref['auc']), abs(C.brier_direct(y, p, ww) - ref['brier']), abs(C.logloss_direct(y, p, ww) - ref['logloss']))
        rv = C.read_json(K.FC / 'eval' / 'results_v.json')['results']['MAIN_TEST']['bucket']['raw']['overall']
        rec['W_A_equals_parent_results_v'] = max(abs(w['sets']['MAIN_TEST']['A_raw']['match_weighted'][k] - rv[k]) for k in ('auc', 'brier', 'logloss'))
        eps = np.finfo(np.float64).eps
        dl = -(y * np.log(np.clip(pB, eps, 1 - eps)) + (1 - y) * np.log(1 - np.clip(pB, eps, 1 - eps))) + (y * np.log(np.clip(pA, eps, 1 - eps)) + (1 - y) * np.log(1 - np.clip(pA, eps, 1 - eps)))
        u, inv_ = np.unique(g, return_inverse=True)
        rows_of = [[] for _ in range(len(u))]
        for i, j in enumerate(inv_.tolist()):
            rows_of[j].append(i)
        rng = np.random.default_rng(K.BOOT_SEED)
        naive = []
        for _ in range(5):
            pick = rng.integers(0, len(u), size=len(u))
            num = sum(dl[rows_of[j]].mean() for j in pick)
            naive.append(num / len(pick))
        first5 = w['sets']['MAIN_TEST']['bootstrap_B_minus_A']['results']['match_weighted']['logloss']['first5']
        rec['bootstrap_W_first5_match_logloss'] = max(abs(a - b) for a, b in zip(naive, first5))
        per = C.read_json(K.OUT / 'results' / 'per_set' / 'MAIN_TEST.json')
        with np.load(K.OUT / 'labels' / 'MAIN_TEST_B_noobj_labels.npz', allow_pickle=False) as z:
            m = (z['valid_h90'] == 1) & (z['cohort'] == 1)
            gg = z['match'][m].astype(str)
            dis = (z['Y_h90_A'][m] != z['Y_h90_B'][m]).astype(float)
        uu, ii = np.unique(gg, return_inverse=True)
        cellT = per['agreement']['h90']['T']
        rec['agreement_T_row'] = abs(dis.mean() - cellT['disagreement_row'])
        rec['agreement_T_match'] = abs((np.bincount(ii, weights=dis) / np.bincount(ii)).mean() - cellT['disagreement_match_weighted'])
        nb = A.naive_bootstrap_replicates(gg, dis, 5, K.BOOT_SEED)
        dr = per['bootstrap_h90']['T']['_draws_first5']['A_vs_B_noobj']
        rec['bootstrap_labels_first5'] = max(max(abs(nb[r][0] - dr['row'][r]), abs(nb[r][1] - dr['match_weighted'][r])) for r in range(5))
        qd = C.read_json(K.OUT / 'results' / 'q_target_dependence.json')
        rec['q_A_reproduction_max'] = max(v['A_reproduction_max_abs_diff'] for v in qd['cohorts'].values())
        rec['W_primary_identity_all_sets'] = all(s['primary_identity']['A_bitwise_equal_parent'] for s in w['sets'].values() if 'primary_identity' in s)
        check('independent_metric_bootstrap_and_reproduction', all(v <= 1e-9 for v in rec.values() if isinstance(v, float)) and rec['W_primary_identity_all_sets'], detail=rec)
        unk_w = {n: s['unknown_objective_team_count']['nonzero_row_share'] for n, s in w['sets'].items() if 'unknown_objective_team_count' in s}
        unk_l = {n: s['unknown_objective_team_count_pre']['nonzero_row_share'] for n, s in lm['summaries'].items()}
        check('unknown_objective_team_count_prevalence_reported', len(unk_w) == 9 and len(unk_l) == 7 and
              vm['unknown_objective_team_count_TRAIN_bucket']['rows'] == 424160, w_bucket=unk_w, engagement_pre=unk_l,
              train_bucket=vm['unknown_objective_team_count_TRAIN_bucket'])
        check('no_human_review_artifacts_created', not any(p.name.lower().startswith(('review', 'private')) for p in K.OUT.iterdir()))
        stat = C.read_json(K.OUT / 'status.json')
        bad = {g_: s['state'] for g_, s in stat['groups'].items() if g_ not in ('postrun_checks', 'report') and s.get('state') != 'complete'}
        check('all_stage_groups_complete', not bad, groups={g_: s.get('state') for g_, s in stat['groups'].items()})
        n_pass = sum(v['pass_'] for v in CHECKS.values())
        out = dict(role=K.ROLE_TAG, version=K.VERSION, checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), checks=CHECKS, passed=n_pass, total=len(CHECKS),
                   all_pass=n_pass == len(CHECKS), scope='computation, identity, gating and reproducibility checks; not semantic or causal validation')
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
