"""Stage Z: freeze all 32 selections (2 cohorts x 2 families x 8 arms) before any TEST/external array or parent TEST prediction is
opened. Reproduces the selection rule from saved Q_SELECT predictions, rebuilds every arm matrix and compares its hash with the fit
record, reloads every chosen bundle and compares Q_CAL/Q_SELECT predictions exactly, checks no sealed access, records parent hashes."""
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
import cc20260916_common as CC  # noqa: E402
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
    CC.log_command()
    base = CC.SMOKE if args.smoke else CC.OUT
    st = CC.Status('freeze' + ('_smoke' if args.smoke else ''))
    try:
        if CC.frozen_path(base).exists():
            raise SystemExit('frozen manifest exists (never overwritten)')
        if not args.smoke and (CC.OUT / 'eval').exists():
            raise SystemExit('eval/ exists before freeze')
        keys = [f'{f}_{a}_{c}' for c in CC.COHORTS for f in CC.FAMILIES for a in CC.ARMS]
        missing = [k for k in keys if not (base / 'selection' / f'{k}.json').exists()]
        if missing:
            raise SystemExit(f'selections missing: {missing}')
        access = [json.loads(l) for l in (base / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (base / 'access_log.jsonl').exists() else []
        sealed_before = [a for a in access if a.get('sealed')]
        if sealed_before:
            raise SystemExit(f'sealed access before freeze: {len(sealed_before)}')
        schema = C.read_json(CC.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        proto = C.read_json(CC.OUT / 'protocol.json')
        reg = proto['fixed_configurations']['registry']
        vocab = list(proto['identity_vocab']['ids'])
        if C.sha256_file(CC.TAG_TABLE) != proto['class_source']['tag_table_sha256']:
            raise SystemExit('tag table differs from the protocol')
        tab = CC.load_tag_table()
        sels, checks, selected, overall, bundle_sha, pred_sha, stop_sha = {}, {}, {}, {}, {}, {}, {}
        for coh in CC.COHORTS:
            st.update('running', f'verify_{coh}', next_step='rebuild arm matrices, reload selected bundles')
            D = CC.load_trainval(base, smoke=args.smoke, cohort=coh)
            names = D['names']
            if names != schema['input_names_all']:
                raise SystemExit('schema drift')
            M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
            ms = M['Q_SELECT']
            ws = CC.weights(D['g'][ms])
            allmet = {}
            for arm in CC.ARMS:
                Xa, names_arm = CC.arm_matrix(D['X'], names, arm, ridge, D['version'], vocab, tab)
                msha = {k: CC.matrix_sha(Xa[m]) for k, m in M.items()}
                for fam in CC.FAMILIES:
                    key = f'{fam}_{arm}_{coh}'
                    sel = C.read_json(base / 'selection' / f'{key}.json')
                    cfg = sel['config']
                    if cfg != reg[coh][fam] or sel['arm'] != arm:
                        raise SystemExit(f'{key}: configuration {cfg} / arm differs from the protocol registry')
                    sels[key] = dict(sha256=C.sha256_file(base / 'selection' / f'{key}.json'), chosen=sel['chosen'], written_at=sel['written_at'])
                    ck = {}
                    p = base / 'models' / coh / fam / arm / f'{cfg}.joblib'
                    bundle_sha[key] = C.sha256_file(p)
                    ck['bundle_hash'] = bundle_sha[key] == sel['bundle_sha256'][cfg]
                    pp = base / 'predictions' / f'{key}_trainval.npz'
                    pred_sha[key] = C.sha256_file(pp)
                    ck['predictions_hash'] = pred_sha[key] == sel['predictions_sha256']
                    ck['arm_matrix_rebuilt_equal'] = msha == sel['arm_matrix_sha256'] and sel['input_names_sha256'] == C.sha256_json(names_arm)
                    with np.load(pp, allow_pickle=False) as z:
                        Z = {k: z[k] for k in z.files}
                    ck['prediction_rows_match_data_keys'] = bool(np.array_equal(Z['match'].astype(str), D['g']) and np.array_equal(Z['s_ms'], D['s_ms'])
                                                                 and np.array_equal(Z['y'], D['y']) and np.array_equal(Z['role'].astype(str), D['role']))
                    direct = {c: (C.brier_direct(D['y'][ms], Z[c][ms], ws), C.logloss_direct(D['y'][ms], Z[c][ms], ws)) for c in CC.candidate_names(cfg) if sel['eligible'][c]}
                    ch_direct, rank_direct = CC.select_rule(direct)
                    ck['selection_rule_reproduced_direct_metrics'] = ch_direct == sel['chosen']
                    ck['max_abs_select_metric_diff_lt_1e-12'] = max(max(abs(direct[c][0] - sel['select_metrics'][c]['brier']), abs(direct[c][1] - sel['select_metrics'][c]['logloss'])) for c in direct) < 1e-12
                    cal = CC.split_candidate(sel['chosen'])[1]
                    b = joblib.load(p)
                    ck['bundle_arm_and_inputs'] = b['arm'] == arm and b['input_names'] == names_arm and b['config'] == cfg
                    for role in ('Q_CAL', 'Q_SELECT'):
                        m = M[role]
                        ck[f'reload_selected_identical_{role}'] = bool(np.array_equal(CC.bundle_predict(b, cal, Xa[m], names_arm), Z[sel['chosen']][m]))
                    b = None
                    ck['reload_identity_at_fit'] = all(sel['reload_identity'][cfg].values())
                    ck['oof_provenance_all_rows'] = sel['oof_provenance']['rows'] == sel['oof_provenance']['adapter_id_equals_own_heldout_fold'] == sel['oof_provenance']['adapter_sha_equals_manifest_oof_hash']
                    if (base / 'internal_stop' / f'{key}.json').exists():
                        stop_sha[key] = C.sha256_file(base / 'internal_stop' / f'{key}.json')
                    checks[key] = ck
                    selected[key] = dict(family=fam, arm=arm, cohort=coh, chosen=sel['chosen'], config=cfg, calibration=cal, n_inputs=sel['n_inputs'],
                                         bundle=f'models/{coh}/{fam}/{arm}/{cfg}.joblib', bundle_sha256=bundle_sha[key],
                                         select_brier=sel['select_metrics'][sel['chosen']]['brier'], select_logloss=sel['select_metrics'][sel['chosen']]['logloss'])
                    for c in CC.candidate_names(cfg):
                        if sel['eligible'][c]:
                            allmet[f'{fam}_{arm}:{c}'] = (sel['select_metrics'][c]['brier'], sel['select_metrics'][c]['logloss'])
                    Z = None
                    st.log(f'{key}: checks {all(ck.values())} chosen {sel["chosen"]} select_brier {selected[key]["select_brier"]:.6f}')
                Xa = None
            ow, orank = CC.select_rule(allmet)
            overall[coh] = dict(chosen=ow, family_arm=ow.split(':')[0], ranking_all_48=orank, rule='same rule over the 48 joint candidates of this cohort (descriptive)')
            D = None
        failed = [f'{k}:{n}' for k, ck in checks.items() for n, ok in ck.items() if not ok]
        if failed:
            raise SystemExit(f'pre-freeze checks failed: {failed}')
        parent_sha = {}
        if not args.smoke:
            for set_name in CC.EVAL_SETS:
                for coh in CC.COHORTS:
                    p = CC.parent_pred_path(set_name, coh)
                    if not p.exists():
                        raise SystemExit(f'parent predictions missing: {p}')
                    parent_sha[f'{set_name}_{coh}'] = C.sha256_file(p)
        iq_fz = C.read_json(Q.frozen_path(CC.IQ))
        ct = CC.OUT / 'contract_tests' / 'result.json'
        frozen = dict(role=CC.ROLE_TAG, version=CC.VERSION, smoke_train_only=bool(args.smoke),
                      protocol_sha256=C.sha256_file(CC.OUT / 'protocol.json'), contract_tests_result_sha256=C.sha256_file(ct) if ct.exists() else None,
                      smoke_frozen_manifest_sha256=(C.sha256_file(CC.frozen_path(CC.SMOKE)) if (not args.smoke and CC.frozen_path(CC.SMOKE).exists()) else None),
                      tag_table_sha256=C.sha256_file(CC.TAG_TABLE), identity_vocab_sha256=proto['identity_vocab']['sha256'],
                      selections=sels, family_winners=selected, overall_winner=overall, bundle_sha256=bundle_sha, trainval_predictions_sha256=pred_sha, internal_stop_sha256=stop_sha,
                      parent_references=dict(iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(CC.IQ)), cohort_role_frozen_manifest_sha256=C.sha256_file(CC.CR / 'frozen_manifest.json'),
                                             iq_winners={k: v['chosen'] for k, v in iq_fz['family_winners'].items()}, parent_predictions_sha256=parent_sha,
                                             note='file hashes only; the parent arrays are opened after this freeze'),
                      pre_freeze_checks=checks, sealed_accesses_before_freeze=len(sealed_before), source_sha256=CC.own_source_hashes(),
                      frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                      statement='All 2 cohorts x 2 families x 8 arms are frozen here (configurations fixed at the iq h90 winners; calibration selected on Q_SELECT). '
                                'TEST/external arrays and parent TEST predictions are opened only after this file exists; no refit or reselection afterwards.')
        frozen['frozen_files_sha256'] = {**{v['bundle']: v['bundle_sha256'] for v in selected.values()},
                                        **{f'selection/{k}.json': v['sha256'] for k, v in sels.items()},
                                        **{f'predictions/{k}_trainval.npz': v for k, v in pred_sha.items()},
                                        **{f'internal_stop/{k}.json': v for k, v in stop_sha.items()}, 'protocol.json': frozen['protocol_sha256']}
        sha = C.write_json(CC.frozen_path(base), frozen)
        st.update('complete', 'frozen', frozen_manifest_sha256=sha, overall={k: v['chosen'] for k, v in overall.items()}, next_step='sealed evaluation')
        return 0
    except SystemExit as exc:
        CC.log_failure(st.group, exc)
        st.update('failed', 'freeze', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        CC.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'freeze', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
