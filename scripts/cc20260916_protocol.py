"""Stage P: write protocol.json before any smoke or full fit (fixed h90 configurations, arms and blocks, pinned tag table and
identity vocabulary, counts, contrasts and the pre-registered primary comparison)."""
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
import cc20260916_common as CC  # noqa: E402


def main():
    CC.log_command()
    if (CC.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json exists (never overwritten)')
    for d in ('models', 'selection', 'smoke_train_only', 'eval'):
        if (CC.OUT / d).exists():
            raise SystemExit(f'{d}/ exists: protocol must predate every fit')
    snap = CC.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        raise SystemExit('integrity snapshot_before.json must precede the protocol')
    schema = C.read_json(CC.FC / 'q_pre_only_schema.json')
    ridge = list(schema['predictor_sets']['ridge'])
    names = list(schema['input_names_all'])
    if len(ridge) != CC.EXPECTED_RIDGE_COUNT or C.q_feature_sets(names)['ridge'] != ridge:
        raise SystemExit('schema contract failed')
    for i in range(10):
        if f'participant_slot{i}_champion_id' not in names or any(f'participant_slot{i}_{s}' not in ridge for s in CC.SLOT_STATS):
            raise SystemExit('slot columns missing from the parent schema')
    cm = C.read_json(CC.CR / 'cohorts' / 'cohort_manifest.json')
    hk = f'h{CC.H}'
    counts = {'TRAIN': {c: sum(cm['sets']['MAIN_TRAIN']['by_sub_role'][f'fold{k}'][hk][c] for k in range(5)) for c in CC.COHORTS},
              'Q_CAL': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_CAL'][hk][c] for c in CC.COHORTS},
              'Q_SELECT': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_SELECT'][hk][c] for c in CC.COHORTS},
              'MAIN_TEST': {c: cm['sets']['MAIN_TEST']['by_sub_role']['TEST'][hk][c] for c in CC.COHORTS}}
    for s in CC.EXT:
        counts[f'EXT_{s}'] = {c: cm['sets'][f'EXT_{s}']['by_sub_role']['EXTERNAL'][hk][c] for c in CC.COHORTS}
    if counts != CC.EXPECTED_COUNTS:
        raise SystemExit(f'cohort manifest counts differ from the specification: {counts}')
    reg, reg_src = CC.registry()
    tab = CC.load_tag_table()
    if not set(CC.SET_VERSION.values()) <= set(tab['table']):
        raise SystemExit('tag table lacks a required version')
    vocab = CC.identity_vocab_from_train(CC.OUT, 'protocol: identity vocabulary from TRAIN')
    missing = [v for v in vocab if str(v) not in tab['table']['15.14.1']]
    if missing or len(vocab) != 171:
        raise SystemExit(f'identity vocabulary not covered by the 15.14.1 table: {missing[:10]} (n={len(vocab)})')
    fetch = C.read_json(CC.DDRAGON / 'fetch_manifest.json')
    before = C.read_json(snap)
    parent_expected = dict(before['sha256_full_corpus_read_targets'])
    parent_expected.update({k: v for k, v in before['sha256_incremental_q'].items() if k.endswith(('frozen_manifest.json', 'protocol.json')) or '/eval/' in k or '/selection/' in k})
    parent_expected.update({k: v for k, v in before['sha256_cohort_role'].items() if '/cohorts/' in k or k.endswith('frozen_manifest.json')})
    arms = {arm: dict(blocks=list(CC.ARM_BLOCKS[arm]), inputs='blocks only (no state, no time, no p_pre)' if arm in CC.DRAFT_ARMS else '352 ridge + blocks',
                      width=CC.arm_width(arm, len(vocab))) for arm in CC.ARMS}
    proto = dict(
        version=CC.VERSION, role=CC.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        design='docs/CLAUDE_CHAMPION_CLASS_20260916.md (Claude, continuing the Codex design line; proposed by the user 2026-09-16)',
        spec_sha256=C.sha256_file(CC.SPEC), provenance_docs_sha256={p.name: C.sha256_file(p) for p in CC.PROVENANCE_DOCS},
        prior_test_exposure='Exploratory follow-up; MAIN TEST and the external sets were evaluated before (cohort-role, iq, Track A, horizon sensitivity, balanced SHAP, V mechanism). No choice here uses TEST performance.',
        question='Do Riot champion-class features (per-slot tags, class-conditioned team state, class composition, position-pair class combinations) lower the sealed Brier of the fixed-configuration logistic / LightGBM q, and beyond champion identity?',
        parents=dict(read_only=['outputs/full_corpus_training_20260915', 'outputs/cohort_role_training_20260915', 'outputs/incremental_q_training_20260915',
                                'source implementations', 'raw data', 'original manuscript'],
                     integrity_snapshot_before_sha256=C.sha256_file(snap), **reg_src,
                     cohort_role_frozen_manifest_sha256=C.sha256_file(CC.CR / 'frozen_manifest.json'),
                     cohort_manifest_sha256=C.sha256_file(CC.CR / 'cohorts' / 'cohort_manifest.json'),
                     q_pre_only_schema_sha256=C.sha256_file(CC.FC / 'q_pre_only_schema.json'),
                     v_models_manifest_sha256=C.sha256_file(CC.FC / 'v_models_manifest.json'),
                     expected_parent_file_sha256=parent_expected),
        class_source=dict(tag_table=CC.TAG_TABLE.relative_to(CC.ROOT).as_posix(), tag_table_sha256=C.sha256_file(CC.TAG_TABLE),
                          fetch_manifest_sha256=C.sha256_file(CC.DDRAGON / 'fetch_manifest.json'), source=fetch['source'], fetched_at=fetch['fetched_at'],
                          approval=fetch['approved_by'], credentials=fetch['credentials'], file_sha256={v: f['sha256'] for v, f in fetch['files'].items()},
                          set_version=dict(CC.SET_VERSION), tags=list(CC.TAGS), n_champions=tab['n_champions'], excluded_entries=tab['excluded_entries'],
                          rule=tab['rule']),
        identity_vocab=dict(source='MAIN_TRAIN h90-valid cohort rows (both cohorts), champion ids > 0, sorted', ids=vocab, n=len(vocab), sha256=C.sha256_json(vocab),
                            unseen_policy='ids outside the vocabulary (external sets) are all-zero rows of the identity block'),
        features=dict(positions=dict(zip(range(5), CC.POSITIONS)), teams={t: list(s) for t, s in CC.TEAMS}, slot_stats=list(CC.SLOT_STATS), agg_names=list(CC.AGG_NAMES),
                      same_team_pairs=[list(p) for p in CC.SAME_TEAM_PAIRS], cross_team_matchups='same position blue vs red', class_membership='primary tag only for counts, aggregates and pairs; secondary tag only in the tags block',
                      block_width=dict(CC.BLOCK_WIDTH, identity=10 * len(vocab)), arms=arms, block_names_sha256={b: C.sha256_json(CC.block_names(b, vocab)) for b in CC.BLOCKS}),
        data_contract=dict(corpus='MAIN 210000 matches: TRAIN 15.14, VALIDATION 15.15 (Q_CAL / Q_SELECT), TEST 15.16; external KR_16.13, KR_16.14_pilot, KR_16.15, NA1_16.13 separately',
                           labels='Y_h90 = 1[delta_h90 > 0] from parent labels unchanged; TRAIN OOF V', cohorts='T: known min >= 4; N: known min < 4',
                           expected_counts=CC.EXPECTED_COUNTS, count_mismatch_policy='blocks fitting / evaluation'),
        inputs=dict(ridge_names_sha256=C.sha256_json(ridge), input_names_all_sha256=C.sha256_json(names), ridge_count=len(ridge)),
        fixed_configurations=dict(registry=reg, source='h90 family winners of incremental_q_training_20260915 (logit, lgbm), identical for every arm',
                                  rule='configuration fixed; only the calibration (raw / sigmoid / isotonic) is selected per family, cohort and arm on Q_SELECT',
                                  stopping='LightGBM iterations re-determined per arm by the unchanged stop10 rule (part of fitting)', lgbm_fixed=Q.LGBM_FIXED,
                                  seeds={f: list(s) for f, s in CC.SEEDS.items()}, limitation='no retuning for wider inputs (stated limitation)'),
        calibration_selection=dict(calibrators='cr20260915_common.fit_calibrators on cohort Q_CAL (raw, sigmoid without positive-slope constraint, isotonic clip)',
                                   candidates_per_arm=3, rule='Q_SELECT match-weighted Brier, then log loss, then name', overall='same rule over the 48 joint candidates of a cohort (descriptive)',
                                   freeze='one frozen_manifest.json for all 32 selections before any TEST/external array or parent TEST prediction is opened'),
        evaluation=dict(sets=list(CC.EVAL_SETS), cells='all, B40, B45, time bins, B40 x time (frozen p_pre and pre time)', named=list(CC.NAMED), parent_named=list(CC.PARENT_NAMED),
                        parent_source='incremental_q eval/predictions/<set>_h90_<cohort>.npz joined by exact (match, s_ms); metrics only; max |parent winner - base arm| reported',
                        contrasts=[dict(a=a, b=b, label=l) for a, b, l in CC.CONTRASTS],
                        primary=dict(a=CC.PRIMARY[0], b=CC.PRIMARY[1], cohort=CC.PRIMARY[2], cell=CC.PRIMARY[3], metric=CC.PRIMARY[4], set='MAIN_TEST'),
                        bootstrap=dict(replicates=CC.BOOT_REPS, seed=CC.BOOT_SEED, models=list(CC.BOOT_MODELS), where='all / B40 / B45 with >= 30 matches and both classes', time_cells='CI not computed'),
                        multiplicity='no adjustment; 20 planned contrasts, one primary'),
        smoke=dict(rows='TRAIN-only 1/8 matches, pseudo roles; all 2 x 2 x 8 fits, smoke freeze and pseudo Q_SELECT evaluation'),
        deferred=['MLP / GPU arms', 'Track B graph adapter', 'subclass taxonomy (13 wiki classes)', 'changes to V or the labels', 'human review', 'causal reading'],
        environment=CC.environment(), source_sha256=CC.own_source_hashes())
    sha = C.write_json(CC.OUT / 'protocol.json', proto)
    print('protocol sha256', sha, 'vocab', len(vocab), 'arms', {a: v['width'] for a, v in arms.items()})


if __name__ == '__main__':
    main()
