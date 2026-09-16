"""Stage Z: freeze the plan (group map hash, bundles and calibrations, selection tags and sizes) before MAIN TEST or any parent
TEST prediction is opened. No fitting. Verifies that every bundle loads and reproduces a TRAIN-row prediction bitwise on reload."""
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
import bsh20260916_common as B  # noqa: E402
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
    B.log_command()
    base = B.SMOKE if args.smoke else B.OUT
    st = B.Status('freeze' + ('_smoke' if args.smoke else ''))
    try:
        if B.frozen_path(base).exists():
            raise SystemExit('frozen manifest exists (never overwritten)')
        if not (B.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json must precede the freeze')
        if not args.smoke:
            ct = B.OUT / 'contract_tests' / 'result.json'
            if not ct.exists() or not C.read_json(ct).get('passed'):
                raise SystemExit('contract tests must pass before the freeze')
            if not (B.frozen_path(B.SMOKE).exists() and (B.SMOKE / 'shap' / 'shap_summary.json').exists()):
                raise SystemExit('TRAIN-only smoke must complete before the full freeze')
            if (B.OUT / 'shap').exists():
                raise SystemExit('shap/ exists before freeze')
        access = [json.loads(l) for l in (base / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (base / 'access_log.jsonl').exists() else []
        sealed_before = [a for a in access if a.get('sealed')]
        if sealed_before:
            raise SystemExit(f'sealed access before freeze: {len(sealed_before)}')
        proto = C.read_json(B.OUT / 'protocol.json')
        schema = C.read_json(B.FC / 'q_pre_only_schema.json')
        ridge, names = list(schema['predictor_sets']['ridge']), list(schema['input_names_all'])
        gidx = B.group_index(names, ridge, schema['shap_groups'])
        gmap = B.group_map_record(ridge, schema['shap_groups'])
        if gmap != proto['groups']['record']:
            raise SystemExit('group map differs from the protocol')
        F, Lb, Co, _ = B.load_parent_set('MAIN_TRAIN', base, 'freeze: reload check on TRAIN rows')
        if [str(x) for x in F['input_names']] != names:
            raise SystemExit('schema drift')
        cols = [names.index(n) for n in ridge]
        checks, models = {}, {}
        for model in B.MODELS:
            models[model] = {}
            for coh in B.COHORTS:
                rec = proto['models'][model][coh]
                path = B.ROOT / rec['bundle']
                if C.sha256_file(path) != rec['bundle_sha256']:
                    raise SystemExit(f'{model} {coh}: bundle hash changed since the protocol')
                b = joblib.load(path)
                rows = np.flatnonzero((Lb['valid_h90'] == 1) & (Co['cohort'] == B.COHORT_CODE[coh]))[:64]
                Z = F['X_input'][rows][:, cols]
                f = B.model_fn(b, rec['calibration'])
                p1 = f(Z)
                b2 = joblib.load(path)
                p2 = B.model_fn(b2, rec['calibration'])(Z)
                ref = (Q if B.MODEL_SOURCE[model][0] == 'iq' else T).bundle_predict(b, rec['calibration'], F['X_input'][rows], names)
                checks[f'{model}_{coh}'] = dict(reload_bitwise=bool(np.array_equal(p1, p2)), model_fn_equals_bundle_predict=bool(np.array_equal(p1, ref)),
                                                finite=bool(np.isfinite(p1).all()), input_schema=b['input_names_all_sha256'] == C.sha256_json(names))
                models[model][coh] = dict(rec, checked_rows=int(len(rows)))
                b = b2 = None
        failed = [f'{k}:{n}' for k, ck in checks.items() for n, ok in ck.items() if not ok]
        if failed:
            raise SystemExit(f'pre-freeze checks failed: {failed}')
        ct = B.OUT / 'contract_tests' / 'result.json'
        frozen = dict(role=B.ROLE_TAG, version=B.VERSION, smoke_train_only=bool(args.smoke), protocol_sha256=C.sha256_file(B.OUT / 'protocol.json'),
                      contract_tests_result_sha256=C.sha256_file(ct) if ct.exists() else None,
                      smoke_frozen_manifest_sha256=(C.sha256_file(B.frozen_path(B.SMOKE)) if (not args.smoke and B.frozen_path(B.SMOKE).exists()) else None),
                      models=models, groups=gmap, group_indices={g: ix for g, ix in zip(B.GROUPS, gidx)},
                      rows=dict(n_explain=B.SMOKE_EXPLAIN if args.smoke else B.N_EXPLAIN, n_background=B.SMOKE_BG if args.smoke else B.N_BG, n_cases=B.N_CASES,
                                tags=dict(explain=B.TAG_EXPLAIN, background=B.TAG_BG, case=B.TAG_CASE), cells=list(B.CELLS), b40=list(B.B40)),
                      parent_references=dict(iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(B.IQ)), ta_frozen_manifest_sha256=C.sha256_file(T.frozen_path(B.TA)),
                                             eval_predictions_sha256={f'{m}_{c}': C.sha256_file(B.parent_eval_predictions(m, c)[0]) for m in B.MODELS for c in B.COHORTS},
                                             note='file hashes only; the parent evaluation arrays are opened after this freeze'),
                      pre_freeze_checks=checks, sealed_accesses_before_freeze=len(sealed_before), source_sha256=B.own_source_hashes(), frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                      statement='Explained models, calibrations, group map and row-selection rules are frozen here; MAIN TEST and the parent TEST predictions are opened only after this file exists.')
        sha = C.write_json(B.frozen_path(base), frozen)
        st.update('complete', 'frozen', frozen_manifest_sha256=sha, next_step='sealed explanation')
        return 0
    except SystemExit as exc:
        B.log_failure(st.group, exc)
        st.update('failed', 'freeze', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        B.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'freeze', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
