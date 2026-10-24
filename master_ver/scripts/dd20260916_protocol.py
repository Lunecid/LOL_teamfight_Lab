"""Stage P: protocol.json before any re-detection, rebuild or fit: the two definitions, rounding rule, procedure, parent hashes, the fixed
configurations and the pre-registered quantities."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import dd20260916_common as DD  # noqa: E402


def main():
    DD.log_command()
    if (DD.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json exists (never overwritten)')
    for d in ('redetect', 'rebuild', 'labels', 'models', 'selection', 'eval'):
        if (DD.OUT / d).exists():
            raise SystemExit(f'{d}/ exists: protocol must predate every stage')
    snap = DD.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        raise SystemExit('integrity snapshot_before.json must precede the protocol')
    dev, pooled = C.read_json(DD.DEV_SPEC), C.read_json(DD.POOLED_SPEC)
    if round(dev['gap_s'], 1) * 1000 != DD.DEFINITIONS['dev']['TF2_KILL_CLUSTER_GAP_MS'] or round(dev['diameter_u']) != DD.DEFINITIONS['dev']['CLUSTER_MAX_DIAMETER']:
        raise SystemExit('dev constants do not follow the rounding rule applied to spec_15.14.json')
    if round(pooled['gap_s'], 1) * 1000 != DD.DEFINITIONS['frozen']['TF2_KILL_CLUSTER_GAP_MS'] or round(pooled['diameter_u']) != DD.DEFINITIONS['frozen']['CLUSTER_MAX_DIAMETER']:
        raise SystemExit('frozen constants do not equal the rounded pooled spec')
    schema = C.read_json(DD.FC / 'q_pre_only_schema.json')
    ridge = list(schema['predictor_sets']['ridge'])
    if len(ridge) != DD.EXPECTED_RIDGE_COUNT:
        raise SystemExit('schema contract failed')
    reg, winners, reg_src = DD.registry()
    before = C.read_json(snap)
    parent_expected = dict(before['sha256_full_corpus_read_targets'])
    parent_expected.update({k: v for k, v in before['sha256_incremental_q'].items() if k.endswith(('frozen_manifest.json', 'protocol.json')) or '/eval/' in k or '/selection/' in k or '/models/' in k})
    parent_expected.update({k: v for k, v in before['sha256_cohort_role'].items() if '/cohorts/' in k or k.endswith('frozen_manifest.json')})
    parent_expected.update(before['sha256_parent_exposures'])
    proto = dict(
        version=DD.VERSION, role=DD.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        design='docs/CLAUDE_DEFINITION_DEV_20260916.md (Claude, continuing the Codex design line)', spec_sha256=C.sha256_file(DD.SPEC),
        provenance_docs_sha256={p.name: C.sha256_file(p) for p in DD.PROVENANCE_DOCS},
        prior_test_exposure='Exploratory follow-up; MAIN TEST and the external sets were evaluated before under the frozen definition. No choice here uses TEST performance.',
        question='Does re-detecting engagements with G/D estimated on the TRAIN patch alone (development-only definition) change the evaluation population and the h90 q results?',
        definitions=dict(frozen=dict(constants=DD.DEFINITIONS['frozen'], source='preset v3.3 = pooled 15.14-15.16 spec rounded', spec=dict(gap_s=pooled['gap_s'], diameter_u=pooled['diameter_u'], n_matches=pooled['n_matches'], sha256=C.sha256_file(DD.POOLED_SPEC))),
                         dev=dict(constants=DD.DEFINITIONS['dev'], source='config/fight_boundary/spec_15.14.json (TRAIN patch only) rounded', spec=dict(gap_s=dev['gap_s'], diameter_u=dev['diameter_u'], n_matches=dev['n_matches'],
                                  gap_ci_s=dev.get('gap_ci_s'), diameter_ci_u=dev.get('diameter_ci_u'), sha256=C.sha256_file(DD.DEV_SPEC))),
                         rounding=DD.DEV_ROUNDING, unchanged='R 1600 u, B 15 s (TF2_ENGAGE_PRE_KILL_MS 15000), M 2, every other detector switch, B-rule endpoints, V, cohort rule (min cluster participation >= 4)'),
        procedure=dict(redetect='parent detector code path (fc20260915_external_prepare.detect_exposures) on every MAIN match and every external cache; frozen reproduction on the first '
                                f'{DD.FROZEN_CHECK_MATCHES} TRAIN matches must be exact', census='per set: matches / rows changed; parent valid T/N rows affected (unsealed sets before the freeze, sealed sets in the evaluation)',
                       rebuild=f'changed matches only, parent extraction function with dev exposures; labels with the frozen V (OOF fold adapter for TRAIN); cohort from the raw detector counts; fixture of {DD.FIXTURE_MATCHES} TRAIN matches with parent exposures must reproduce the parent rows bitwise',
                       population='dev population = parent rows of unchanged matches (copied) + rebuilt rows; TEST / external populations are assembled only after the freeze',
                       fit='base arm (352 ridge) logit + LightGBM with the iq h90 configurations, calibration selected on dev Q_SELECT, T and N', evaluate='dev MAIN TEST + externals: refit winners and frozen parent winners on the same rows'),
        parents=dict(read_only=['outputs/full_corpus_training_20260915', 'outputs/cohort_role_training_20260915', 'outputs/incremental_q_training_20260915', 'outputs/postkill_objective_delay_full/exposures.csv',
                                'caches', 'original repository detector sources', 'worktree extraction sources'],
                     integrity_snapshot_before_sha256=C.sha256_file(snap), **reg_src, cohort_role_frozen_manifest_sha256=C.sha256_file(DD.CR / 'frozen_manifest.json'),
                     q_pre_only_schema_sha256=C.sha256_file(DD.FC / 'q_pre_only_schema.json'), v_models_manifest_sha256=C.sha256_file(DD.FC / 'v_models_manifest.json'),
                     parent_exposures_sha256=C.sha256_file(C.EXPOSURES), expected_parent_file_sha256=parent_expected, iq_winners=winners),
        data_contract=dict(corpus='MAIN 210000 matches: TRAIN 15.14, VALIDATION 15.15 (Q_CAL / Q_SELECT), TEST 15.16; externals KR_16.13, KR_16.14_pilot, KR_16.15, NA1_16.13',
                           old_population_counts=DD.EXPECTED_COUNTS_OLD, labels='Y_h90 = 1[delta_h90 > 0]; TRAIN OOF V; unchanged V', cohorts='T: known min >= 4; N: known min < 4'),
        inputs=dict(ridge_names_sha256=C.sha256_json(ridge), input_names_all_sha256=C.sha256_json(schema['input_names_all'])),
        fixed_configurations=dict(registry=reg, source='iq h90 family winners (logit, lgbm)', rule='configuration fixed; calibration selected per family and cohort on dev Q_SELECT', lgbm_fixed=Q.LGBM_FIXED, seeds={f: list(s) for f, s in DD.SEEDS.items()}),
        evaluation=dict(sets=list(DD.EVAL_SETS), cells='all, B40, B45, time bins, B40 x time', named=list(DD.NAMED), contrasts=[dict(a=a, b=b, label=l) for a, b, l in DD.CONTRASTS],
                        primary=dict(quantity='cohort T MAIN TEST: frozen parent LightGBM Brier on the dev population vs the parent-reported Brier on the old population, with the fraction of changed T rows',
                                     contrast='refit LightGBM - frozen LightGBM, dev population, all cell, Brier', set='MAIN_TEST', cohort='T'),
                        identity='frozen winner predictions on common rows must equal the parent eval predictions', bootstrap=dict(replicates=DD.BOOT_REPS, seed=DD.BOOT_SEED, where='all / B40 / B45 with >= 30 matches and both classes'),
                        population_comparison='old vs dev population metrics are descriptive (different rows); no interval across definitions'),
        not_done=['G/D re-estimation on 15.14 + 15.15 pooled', 'V changes', 'human review', 'causal reading'],
        environment=DD.environment(), source_sha256=DD.own_source_hashes())
    sha = C.write_json(DD.OUT / 'protocol.json', proto)
    print('protocol sha256', sha, DD.DEFINITIONS)


if __name__ == '__main__':
    main()
