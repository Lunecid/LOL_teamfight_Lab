"""Stage Z: freeze all selections for BOTH cohorts before any TEST/external array (or iq/parent TEST prediction) is opened.

Checks before writing frozen_manifest.json: every family x cohort selection exists; every saved bundle hash matches;
the selection rule is reproduced from the saved Q_SELECT predictions with independent direct metric formulas and
weights recomputed from Q_SELECT match ids; every selected bundle is reloaded and its Q_CAL/Q_SELECT predictions are
recomputed from the parent input arrays and compared exactly; no sealed access so far. Records the iq frozen manifest
and iq evaluation prediction file hashes (bytes only; the arrays stay closed) and a five-family Q_SELECT ranking
(iq PT / logistic / LightGBM winners + the two MLP winners) on the identical cohort Q_SELECT rows.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta20260916_common as T  # noqa: E402
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    T.log_command()
    base = T.SMOKE if args.smoke else T.OUT
    st = T.Status('freeze' + ('_smoke' if args.smoke else ''))
    try:
        if T.frozen_path(base).exists():
            raise SystemExit(f'{T.frozen_path(base)} exists (never overwritten)')
        if not args.smoke and (T.OUT / 'eval').exists():
            raise SystemExit('eval/ exists before freeze')
        missing = [f'{f}_{c}' for c in T.COHORTS for f in T.FAMILIES if not (base / 'selection' / f'{f}_{c}.json').exists()]
        if missing:
            raise SystemExit(f'selections missing: {missing}')
        access = []
        if (base / 'access_log.jsonl').exists():
            access = [json.loads(l) for l in (base / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        sealed_before = [a for a in access if a.get('sealed')]
        if sealed_before:
            raise SystemExit(f'sealed access before freeze: {len(sealed_before)}')
        schema = C.read_json(T.FC / 'q_pre_only_schema.json')
        sels, checks, selected, overall, bundle_sha, pred_sha, five = {}, {}, {}, {}, {}, {}, {}
        for coh in T.COHORTS:
            st.update('running', f'verify_{coh}', next_step='reload selected bundles')
            D = T.load_trainval(base, smoke=args.smoke, cohort=coh)
            names = D['names']
            if names != schema['input_names_all']:
                raise SystemExit('schema drift')
            allmet = {}
            for fam in T.FAMILIES:
                key = f'{fam}_{coh}'
                sel = C.read_json(base / 'selection' / f'{key}.json')
                sels[key] = dict(sha256=C.sha256_file(base / 'selection' / f'{key}.json'), chosen=sel['chosen'], written_at=sel['written_at'])
                ck = {}
                bundle_sha[key] = {}
                for cfg, sha in sel['bundle_sha256'].items():
                    p = base / 'models' / coh / fam / f'{cfg}.joblib'
                    bundle_sha[key][cfg] = C.sha256_file(p)
                    ck[f'bundle_hash_{cfg}'] = bundle_sha[key][cfg] == sha
                pp = base / 'predictions' / f'{fam}_{coh}_trainval.npz'
                pred_sha[key] = C.sha256_file(pp)
                ck['predictions_hash'] = pred_sha[key] == sel['predictions_sha256']
                with np.load(pp, allow_pickle=False) as z:
                    Z = {k: z[k] for k in z.files}
                ck['prediction_rows_match_data_keys'] = bool(np.array_equal(Z['match'].astype(str), D['g']) and np.array_equal(Z['s_ms'], D['s_ms'])
                                                             and np.array_equal(Z['y'], D['y']) and np.array_equal(Z['role'].astype(str), D['role']))
                ms = D['role'] == 'Q_SELECT'
                ws = T.weights(D['g'][ms])
                direct = {c: (C.brier_direct(D['y'][ms], Z[c][ms], ws), C.logloss_direct(D['y'][ms], Z[c][ms], ws))
                          for c in T.candidate_names(fam) if sel['eligible'][c]}
                ch_direct, rank_direct = T.select_rule(direct)
                ck['selection_rule_reproduced_direct_metrics'] = ch_direct == sel['chosen']
                ck['INFO_full_ranking_reproduced_direct_metrics'] = rank_direct == sel['ranking']
                ck['max_abs_select_metric_diff_lt_1e-12'] = max(max(abs(direct[c][0] - sel['select_metrics'][c]['brier']),
                                                                    abs(direct[c][1] - sel['select_metrics'][c]['logloss'])) for c in direct) < 1e-12
                cfg, cal = T.split_candidate(sel['chosen'])
                b = joblib.load(base / 'models' / coh / fam / f'{cfg}.joblib')
                for role in ('Q_CAL', 'Q_SELECT'):
                    m = D['role'] == role
                    ck[f'reload_selected_identical_{role}'] = bool(np.array_equal(T.bundle_predict(b, cal, D['X'][m], names), Z[sel['chosen']][m]))
                b = None
                ck['all_configs_reload_identity_at_fit'] = all(all(v.values()) for v in sel['reload_identity'].values())
                ck['oof_provenance_all_rows'] = sel['oof_provenance']['rows'] == sel['oof_provenance']['adapter_id_equals_own_heldout_fold'] == \
                    sel['oof_provenance']['adapter_sha_equals_manifest_oof_hash']
                checks[key] = ck
                selected[key] = dict(family=fam, cohort=coh, chosen=sel['chosen'], config=cfg, calibration=cal,
                                     bundle=f'models/{coh}/{fam}/{cfg}.joblib', bundle_sha256=bundle_sha[key][cfg],
                                     select_brier=sel['select_metrics'][sel['chosen']]['brier'], select_logloss=sel['select_metrics'][sel['chosen']]['logloss'])
                for c in T.candidate_names(fam):
                    if sel['eligible'][c]:
                        allmet[c] = (sel['select_metrics'][c]['brier'], sel['select_metrics'][c]['logloss'])
                Z = None
            ow, orank = T.select_rule(allmet)
            winners = {selected[f'{f}_{coh}']['chosen']: allmet[selected[f'{f}_{coh}']['chosen']] for f in T.FAMILIES}
            ow2, _ = T.select_rule(winners)
            overall[coh] = dict(chosen=ow, family=ow.split('_')[0], equals_rule_over_two_family_winners=ow == ow2, ranking_all_36=orank,
                                rule='same rule (Q_SELECT Brier, log loss, name) over all eligible joint candidates of the two MLP families')
            # five-family ranking on the identical cohort Q_SELECT rows (iq winners frozen; TRAIN/VALIDATION predictions are not sealed)
            if not args.smoke:
                fam5 = {}
                same_rows = {}
                for iqf in T.IQ_FAMILIES:
                    isel = C.read_json(T.IQ / 'selection' / f'{iqf}_{coh}.json')
                    with np.load(T.IQ / 'predictions' / f'{iqf}_{coh}_trainval.npz', allow_pickle=False) as z:
                        same_rows[iqf] = bool(np.array_equal(z['match'].astype(str), D['g']) and np.array_equal(z['s_ms'], D['s_ms'])
                                              and np.array_equal(z['role'].astype(str), D['role']) and np.array_equal(z['y'], D['y']))
                        ip = z[isel['chosen']][ms]
                    fam5[f'iq_{iqf}:{isel["chosen"]}'] = (C.brier_direct(D['y'][ms], ip, ws), C.logloss_direct(D['y'][ms], ip, ws))
                    if abs(fam5[f'iq_{iqf}:{isel["chosen"]}'][0] - isel['select_metrics'][isel['chosen']]['brier']) > 1e-12:
                        raise SystemExit(f'iq {iqf} {coh} Q_SELECT Brier not reproduced on these rows')
                for f in T.FAMILIES:
                    fam5[f'ta_{f}:{selected[f"{f}_{coh}"]["chosen"]}'] = winners[selected[f'{f}_{coh}']['chosen']]
                if not all(same_rows.values()):
                    raise SystemExit(f'iq TRAIN/VALIDATION rows differ from this run: {same_rows}')
                w5, r5 = T.select_rule(fam5)
                five[coh] = dict(chosen=w5, ranking=r5, metrics={k: dict(brier=v[0], logloss=v[1]) for k, v in fam5.items()},
                                 iq_rows_identical=same_rows, note='information only; iq winners are frozen and not refit')
            D = None
        failed = [f'{k}:{n}' for k, ck in checks.items() for n, ok in ck.items() if not ok and not n.startswith('INFO_')]
        failed += [f'overall_{c}' for c, o in overall.items() if not o['equals_rule_over_two_family_winners']]
        if failed:
            raise SystemExit(f'pre-freeze checks failed: {failed}')
        iq_fz = T.iq_frozen()
        iq_pred_sha = {}
        if not args.smoke:
            for set_name in T.EVAL_SETS:
                for coh in T.COHORTS:
                    p = T.iq_eval_predictions_path(set_name, coh)
                    if not p.exists():
                        raise SystemExit(f'iq evaluation predictions missing: {p}')
                    iq_pred_sha[f'{set_name}_h90_{coh}'] = C.sha256_file(p)
        ct = T.OUT / 'contract_tests' / 'result.json'
        frozen = dict(
            role=T.ROLE_TAG, version=T.VERSION, smoke_train_only=bool(args.smoke),
            protocol_sha256=C.sha256_file(T.OUT / 'protocol.json'),
            contract_tests_result_sha256=C.sha256_file(ct) if ct.exists() else None,
            smoke_frozen_manifest_sha256=(C.sha256_file(T.frozen_path(T.SMOKE)) if (not args.smoke and T.frozen_path(T.SMOKE).exists()) else None),
            selections=sels, family_winners=selected, overall_winner_mlp_families=overall, five_family_q_select_ranking=five,
            bundle_sha256=bundle_sha, trainval_predictions_sha256=pred_sha,
            internal_stop_sha256={f'{f}_{c}': C.sha256_file(base / 'internal_stop' / f'{f}_{c}.json') for f in T.FAMILIES for c in T.COHORTS},
            iq_references=dict(frozen_manifest_sha256=C.sha256_file(Q.frozen_path(T.IQ)),
                               family_winners={k: v['chosen'] for k, v in iq_fz['family_winners'].items()},
                               overall_winner={c: o['chosen'] for c, o in iq_fz['overall_winner'].items()},
                               legacy_references=iq_fz['legacy_references'], eval_predictions_sha256=iq_pred_sha,
                               note='file hashes only; the iq evaluation arrays are opened after this freeze'),
            pre_freeze_checks=checks, sealed_accesses_before_freeze=len(sealed_before),
            source_sha256=T.own_source_hashes(), frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            statement=('All plain MLP / residual MLP configurations, calibrations, family winners and the MLP-family overall winner for T and N '
                       'are frozen here. TEST 15.16 and external arrays and the iq / parent TEST predictions are opened only after this file '
                       'exists; no refit or reselection afterwards. Prior exposure of old TEST results is acknowledged.'))
        frozen['frozen_files_sha256'] = {**{f'models/{c}/{f}/{cfg}.joblib': sha for f in T.FAMILIES for c in T.COHORTS
                                           for cfg, sha in bundle_sha[f'{f}_{c}'].items()},
                                        **{f'selection/{k}.json': v['sha256'] for k, v in sels.items()},
                                        **{f'predictions/{k}_trainval.npz': v for k, v in pred_sha.items()},
                                        **{f'internal_stop/{k}.json': v for k, v in frozen['internal_stop_sha256'].items()},
                                        'protocol.json': frozen['protocol_sha256']}
        sha = C.write_json(T.frozen_path(base), frozen)
        st.update('complete', 'frozen', frozen_manifest_sha256=sha, winners={k: v['chosen'] for k, v in selected.items()},
                  overall={c: o['chosen'] for c, o in overall.items()}, five_family={c: v['chosen'] for c, v in five.items()},
                  next_step='sealed evaluation' + (' (smoke pseudo rows)' if args.smoke else ''))
        return 0
    except SystemExit as exc:
        T.log_failure(st.group, exc)
        st.update('failed', 'freeze', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        T.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'freeze', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
