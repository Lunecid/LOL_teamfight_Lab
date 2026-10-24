"""Stage Z: freeze all selections (both horizons x both cohorts x five families) before any TEST/external array or parent TEST
prediction is opened. Reproduces the selection rule from saved Q_SELECT predictions, reloads every chosen bundle and compares its
Q_CAL/Q_SELECT predictions exactly, checks no sealed access, records parent frozen-manifest and legacy prediction file hashes."""
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
import hs20260916_common as HS  # noqa: E402
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402

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
    HS.log_command()
    base = HS.SMOKE if args.smoke else HS.OUT
    horizons = (60,) if args.smoke else HS.HORIZONS
    st = HS.Status('freeze' + ('_smoke' if args.smoke else ''))
    try:
        if HS.frozen_path(base).exists():
            raise SystemExit('frozen manifest exists (never overwritten)')
        if not args.smoke and (HS.OUT / 'eval').exists():
            raise SystemExit('eval/ exists before freeze')
        missing = [f'h{h}_{f}_{c}' for h in horizons for c in HS.COHORTS for f in HS.FAMILIES if not (base / 'selection' / f'h{h}_{f}_{c}.json').exists()]
        if missing:
            raise SystemExit(f'selections missing: {missing}')
        access = [json.loads(l) for l in (base / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (base / 'access_log.jsonl').exists() else []
        sealed_before = [a for a in access if a.get('sealed')]
        if sealed_before:
            raise SystemExit(f'sealed access before freeze: {len(sealed_before)}')
        schema = C.read_json(HS.FC / 'q_pre_only_schema.json')
        proto = C.read_json(HS.OUT / 'protocol.json')
        reg = proto['fixed_configurations']['registry']
        sels, checks, selected, overall, bundle_sha, pred_sha, stop_sha = {}, {}, {}, {}, {}, {}, {}
        for h in horizons:
            for coh in HS.COHORTS:
                st.update('running', f'verify_h{h}_{coh}', next_step='reload selected bundles')
                D = HS.load_trainval(base, h, smoke=args.smoke, cohort=coh)
                names = D['names']
                if names != schema['input_names_all']:
                    raise SystemExit('schema drift')
                ms = D['role'] == 'Q_SELECT'
                ws = HS.weights(D['g'][ms])
                allmet = {}
                for fam in HS.FAMILIES:
                    key = f'h{h}_{fam}_{coh}'
                    sel = C.read_json(base / 'selection' / f'{key}.json')
                    cfg = sel['config']
                    if cfg != reg[coh][fam]:
                        raise SystemExit(f'{key}: configuration {cfg} differs from the protocol registry {reg[coh][fam]}')
                    sels[key] = dict(sha256=C.sha256_file(base / 'selection' / f'{key}.json'), chosen=sel['chosen'], written_at=sel['written_at'])
                    ck = {}
                    p = base / 'models' / f'h{h}' / coh / fam / f'{cfg}.joblib'
                    bundle_sha[key] = C.sha256_file(p)
                    ck['bundle_hash'] = bundle_sha[key] == sel['bundle_sha256'][cfg]
                    pp = base / 'predictions' / f'{key}_trainval.npz'
                    pred_sha[key] = C.sha256_file(pp)
                    ck['predictions_hash'] = pred_sha[key] == sel['predictions_sha256']
                    with np.load(pp, allow_pickle=False) as z:
                        Z = {k: z[k] for k in z.files}
                    ck['prediction_rows_match_data_keys'] = bool(np.array_equal(Z['match'].astype(str), D['g']) and np.array_equal(Z['s_ms'], D['s_ms'])
                                                                 and np.array_equal(Z['y'], D['y']) and np.array_equal(Z['role'].astype(str), D['role']))
                    direct = {c: (C.brier_direct(D['y'][ms], Z[c][ms], ws), C.logloss_direct(D['y'][ms], Z[c][ms], ws)) for c in HS.candidate_names(cfg) if sel['eligible'][c]}
                    ch_direct, rank_direct = HS.select_rule(direct)
                    ck['selection_rule_reproduced_direct_metrics'] = ch_direct == sel['chosen']
                    ck['max_abs_select_metric_diff_lt_1e-12'] = max(max(abs(direct[c][0] - sel['select_metrics'][c]['brier']), abs(direct[c][1] - sel['select_metrics'][c]['logloss'])) for c in direct) < 1e-12
                    cal = HS.split_candidate(sel['chosen'])[1]
                    b = joblib.load(p)
                    for role in ('Q_CAL', 'Q_SELECT'):
                        m = D['role'] == role
                        ck[f'reload_selected_identical_{role}'] = bool(np.array_equal(HS.bundle_predict(b, cal, D['X'][m], names), Z[sel['chosen']][m]))
                    b = None
                    ck['reload_identity_at_fit'] = all(sel['reload_identity'][cfg].values())
                    ck['oof_provenance_all_rows'] = sel['oof_provenance']['rows'] == sel['oof_provenance']['adapter_id_equals_own_heldout_fold'] == sel['oof_provenance']['adapter_sha_equals_manifest_oof_hash']
                    if (base / 'internal_stop' / f'{key}.json').exists():
                        stop_sha[key] = C.sha256_file(base / 'internal_stop' / f'{key}.json')
                    checks[key] = ck
                    selected[key] = dict(horizon_s=h, family=fam, cohort=coh, chosen=sel['chosen'], config=cfg, calibration=cal,
                                         bundle=f'models/h{h}/{coh}/{fam}/{cfg}.joblib', bundle_sha256=bundle_sha[key],
                                         select_brier=sel['select_metrics'][sel['chosen']]['brier'], select_logloss=sel['select_metrics'][sel['chosen']]['logloss'])
                    for c in HS.candidate_names(cfg):
                        if sel['eligible'][c]:
                            allmet[f'{fam}:{c}'] = (sel['select_metrics'][c]['brier'], sel['select_metrics'][c]['logloss'])
                    Z = None
                ow, orank = HS.select_rule(allmet)
                overall[f'h{h}_{coh}'] = dict(chosen=ow, family=ow.split(':')[0], ranking_all_15=orank, rule='same rule over the 15 joint candidates of this horizon and cohort')
                D = None
        failed = [f'{k}:{n}' for k, ck in checks.items() for n, ok in ck.items() if not ok]
        if failed:
            raise SystemExit(f'pre-freeze checks failed: {failed}')
        fz_cr = C.read_json(HS.CR / 'frozen_manifest.json')
        legacy_sha = {}
        if not args.smoke:
            for h in horizons:
                for set_name in HS.EVAL_SETS:
                    for coh in HS.COHORTS:
                        p = HS.legacy_path(set_name, h, coh)
                        if not p.exists():
                            raise SystemExit(f'legacy predictions missing: {p}')
                        legacy_sha[f'{set_name}_h{h}_{coh}'] = C.sha256_file(p)
        ct = HS.OUT / 'contract_tests' / 'result.json'
        frozen = dict(role=HS.ROLE_TAG, version=HS.VERSION, smoke_train_only=bool(args.smoke), horizons=list(horizons),
                      protocol_sha256=C.sha256_file(HS.OUT / 'protocol.json'), contract_tests_result_sha256=C.sha256_file(ct) if ct.exists() else None,
                      smoke_frozen_manifest_sha256=(C.sha256_file(HS.frozen_path(HS.SMOKE)) if (not args.smoke and HS.frozen_path(HS.SMOKE).exists()) else None),
                      selections=sels, family_winners=selected, overall_winner=overall, bundle_sha256=bundle_sha, trainval_predictions_sha256=pred_sha, internal_stop_sha256=stop_sha,
                      parent_references=dict(iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(HS.IQ)), ta_frozen_manifest_sha256=C.sha256_file(T.frozen_path(HS.TA)),
                                             cohort_role_frozen_manifest_sha256=C.sha256_file(HS.CR / 'frozen_manifest.json'),
                                             legacy_A_specialists={k: v['chosen'] for k, v in fz_cr['A_specialists'].items()},
                                             legacy_pooled={hh: v['chosen'] for hh, v in fz_cr['pooled_reference']['q_selections'].items()},
                                             legacy_predictions_sha256=legacy_sha, note='file hashes only; the legacy arrays are opened after this freeze'),
                      pre_freeze_checks=checks, sealed_accesses_before_freeze=len(sealed_before), source_sha256=HS.own_source_hashes(),
                      frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                      statement='All five families x two cohorts x the horizons listed are frozen here (configurations fixed at h90; calibration selected on Q_SELECT). '
                                'TEST/external arrays and parent TEST predictions are opened only after this file exists; no refit or reselection afterwards.')
        frozen['frozen_files_sha256'] = {**{v['bundle']: v['bundle_sha256'] for v in selected.values()},
                                        **{f'selection/{k}.json': v['sha256'] for k, v in sels.items()},
                                        **{f'predictions/{k}_trainval.npz': v for k, v in pred_sha.items()},
                                        **{f'internal_stop/{k}.json': v for k, v in stop_sha.items()}, 'protocol.json': frozen['protocol_sha256']}
        sha = C.write_json(HS.frozen_path(base), frozen)
        st.update('complete', 'frozen', frozen_manifest_sha256=sha, winners={k: v['chosen'] for k, v in selected.items()}, overall={k: v['chosen'] for k, v in overall.items()},
                  next_step='sealed evaluation')
        return 0
    except SystemExit as exc:
        HS.log_failure(st.group, exc)
        st.update('failed', 'freeze', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        HS.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'freeze', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
