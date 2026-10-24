"""Stage P: write protocol.json before any smoke or full fit (fixed h90 configurations, counts per horizon, policies)."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402
import hs20260916_common as HS  # noqa: E402


def main():
    HS.log_command()
    if (HS.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json exists (never overwritten)')
    for d in ('models', 'selection', 'smoke_train_only'):
        if (HS.OUT / d).exists():
            raise SystemExit(f'{d}/ exists: protocol must predate every fit')
    snap = HS.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        raise SystemExit('integrity snapshot_before.json must precede the protocol')
    schema = C.read_json(HS.FC / 'q_pre_only_schema.json')
    ridge = list(schema['predictor_sets']['ridge'])
    names = list(schema['input_names_all'])
    if len(ridge) != HS.EXPECTED_RIDGE_COUNT or C.q_feature_sets(names)['ridge'] != ridge:
        raise SystemExit('schema contract failed')
    cm = C.read_json(HS.CR / 'cohorts' / 'cohort_manifest.json')
    counts = {}
    for h in HS.HORIZONS + (HS.H90,):
        hk = f'h{h}'
        mc = {'TRAIN': {c: sum(cm['sets']['MAIN_TRAIN']['by_sub_role'][f'fold{k}'][hk][c] for k in range(5)) for c in HS.COHORTS},
              'Q_CAL': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_CAL'][hk][c] for c in HS.COHORTS},
              'Q_SELECT': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_SELECT'][hk][c] for c in HS.COHORTS},
              'MAIN_TEST': {c: cm['sets']['MAIN_TEST']['by_sub_role']['TEST'][hk][c] for c in HS.COHORTS}}
        for s in HS.EXT:
            mc[f'EXT_{s}'] = {c: cm['sets'][f'EXT_{s}']['by_sub_role']['EXTERNAL'][hk][c] for c in HS.COHORTS}
        counts[hk] = mc
        if mc != HS.EXPECTED_COUNTS:
            raise SystemExit(f'cohort manifest counts at {hk} differ from the h90 counts used as specification: {mc}')
    reg, reg_src = HS.h90_registry()
    fz_cr = C.read_json(HS.CR / 'frozen_manifest.json')
    before = C.read_json(snap)
    parent_expected = dict(before['sha256_full_corpus_read_targets'])
    for key in ('sha256_incremental_q', 'sha256_track_a'):
        parent_expected.update({k: v for k, v in before[key].items() if k.endswith(('frozen_manifest.json', 'protocol.json')) or '/eval/' in k or '/selection/' in k})
    parent_expected.update({k: v for k, v in before['sha256_cohort_role'].items() if '/cohorts/' in k or k.endswith(('frozen_manifest.json', 'results_A.json'))
                            or ('/eval/predictions/' in k and ('_h60_' in k or '_h120_' in k))})
    arch = {}
    for coh in HS.COHORTS:
        for f in HS.TA_FAMILIES:
            p = T.config_params(f, reg[coh][f])
            arch[reg[coh][f]] = dict(n_params=T.param_count(T.build_module(f, HS.EXPECTED_RIDGE_COUNT, p['width'], p['dropout'])))
    proto = dict(
        version=HS.VERSION, role=HS.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        design='docs/CLAUDE_HORIZON_SENSITIVITY_20260916.md (Claude, continuing the Codex design line)',
        spec_sha256=C.sha256_file(HS.SPEC), provenance_docs_sha256={p.name: C.sha256_file(p) for p in HS.PROVENANCE_DOCS},
        prior_test_exposure='Exploratory follow-up; MAIN TEST and the external sets were evaluated before at h60/h90/h120 (cohort-role) and at h90 (iq, ta). No choice here uses TEST performance.',
        horizons=list(HS.HORIZONS), h90_reference='not refit; the iq and ta h90 results are quoted descriptively',
        parents=dict(read_only=['outputs/full_corpus_training_20260915', 'outputs/cohort_role_training_20260915', 'outputs/incremental_q_training_20260915',
                                'outputs/track_a_mlp_20260916', 'source implementations', 'raw data', 'original manuscript'],
                     integrity_snapshot_before_sha256=C.sha256_file(snap), **reg_src,
                     cohort_role_frozen_manifest_sha256=C.sha256_file(HS.CR / 'frozen_manifest.json'),
                     cohort_manifest_sha256=C.sha256_file(HS.CR / 'cohorts' / 'cohort_manifest.json'),
                     q_pre_only_schema_sha256=C.sha256_file(HS.FC / 'q_pre_only_schema.json'),
                     v_models_manifest_sha256=C.sha256_file(HS.FC / 'v_models_manifest.json'),
                     legacy_A_specialists={k: v['chosen'] for k, v in fz_cr['A_specialists'].items()},
                     legacy_pooled={h: v['chosen'] for h, v in fz_cr['pooled_reference']['q_selections'].items()},
                     expected_parent_file_sha256=parent_expected),
        data_contract=dict(corpus='MAIN 210000 matches: TRAIN 15.14, VALIDATION 15.15 (Q_CAL / Q_SELECT), TEST 15.16; external KR_16.13, KR_16.14_pilot, KR_16.15, NA1_16.13 separately',
                           labels={f'h{h}': f'Y_h{h} = 1[delta_h{h} > 0] from parent labels unchanged; TRAIN OOF V' for h in HS.HORIZONS},
                           endpoint='e(h) = min(last kill + h, next kill - 1 ms, next engagement start - 1 ms, game end - 1 ms) (parent)',
                           cohorts='T: known min >= 4; N: known min < 4', expected_counts_by_horizon=counts, count_mismatch_policy='blocks fitting / evaluation',
                           rows='valid_h{h} and cohort; counts identical at h60 / h90 / h120 per the cohort manifest, so the evaluation rows coincide across horizons'),
        inputs=dict(learners='predictor_sets["ridge"] (352) for logistic / LightGBM / MLPs; (p_pre_V, time_minutes) for PT',
                    ridge_names_sha256=C.sha256_json(ridge), input_names_all_sha256=C.sha256_json(names)),
        fixed_configurations=dict(registry=reg, source='h90 family winners of incremental_q_training_20260915 (pt, logit, lgbm) and track_a_mlp_20260916 (mlp, resmlp)',
                                  rule='configuration fixed; only the calibration (raw / sigmoid / isotonic) is selected per family, cohort and horizon on Q_SELECT',
                                  stopping='LightGBM iterations and MLP epochs re-determined on the horizon labels by the unchanged stop10 rule (part of fitting)',
                                  mlp_architectures=arch, lgbm_fixed=Q.LGBM_FIXED, mlp_optimizer=T.OPT, seeds={f: list(s) for f, s in HS.SEEDS.items()},
                                  prediction_definition_mlp=f'CPU float64, {T.PRED_THREADS} thread, chunk {T.PRED_CHUNK} (as Track A)'),
        calibration_selection=dict(calibrators='cr20260915_common.fit_calibrators on cohort Q_CAL (raw, sigmoid without positive-slope constraint, isotonic clip)',
                                   candidates_per_family=3, rule='Q_SELECT match-weighted Brier, then log loss, then name', overall='same rule over the 15 joint candidates of a horizon and cohort',
                                   freeze='one frozen_manifest.json for both horizons before any TEST/external array or parent TEST prediction is opened'),
        evaluation=dict(sets=list(HS.EVAL_SETS), cells='all, B40, B45, time bins, B40 x time (frozen p_pre and pre time)',
                        named=list(HS.NAMED), legacy_source='cohort_role eval/predictions/A_<set>_h<h>_<cohort>.npz joined by exact (match, s_ms)',
                        contrasts=[dict(a=a, b=b, label=l) for a, b, l in HS.CONTRASTS],
                        bootstrap=dict(replicates=HS.BOOT_REPS, seed=HS.BOOT_SEED, where='all / B40 / B45 with >= 30 matches and both classes', time_cells='CI not computed'),
                        cross_horizon='descriptive only (labels differ); no cross-horizon interval'),
        deferred=['SHAP', 'Track B', 'revised V', 'logit-label decomposition', 'development-only detector re-estimation'],
        environment=HS.environment(), source_sha256=HS.own_source_hashes())
    sha = C.write_json(HS.OUT / 'protocol.json', proto)
    print('protocol sha256', sha)


if __name__ == '__main__':
    main()
