"""Stage P: write protocol.json before the smoke and the freeze (models, group map, selection rules, checks)."""
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
import bsh20260916_common as B  # noqa: E402


def main():
    B.log_command()
    if (B.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json exists (never overwritten)')
    snap = B.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        raise SystemExit('integrity snapshot_before.json must precede the protocol')
    schema = C.read_json(B.FC / 'q_pre_only_schema.json')
    ridge, names = list(schema['predictor_sets']['ridge']), list(schema['input_names_all'])
    gidx = B.group_index(names, ridge, schema['shap_groups'])
    gmap = B.group_map_record(ridge, schema['shap_groups'])
    ta_fz = C.read_json(T.frozen_path(B.TA))
    models = {}
    for model in B.MODELS:
        models[model] = {}
        for coh in B.COHORTS:
            path, cal, sha, chosen = B.parent_winner(model, coh)
            if C.sha256_file(path) != sha:
                raise SystemExit(f'{model} {coh}: bundle hash differs from the parent freeze')
            models[model][coh] = dict(bundle=path.relative_to(B.ROOT).as_posix(), bundle_sha256=sha, calibration=cal, candidate=chosen,
                                      five_family_q_select_winner=ta_fz['five_family_q_select_ranking'][coh]['chosen'])
            if model == 'final_q' and ta_fz['five_family_q_select_ranking'][coh]['chosen'] != f'iq_lgbm:{chosen}':
                raise SystemExit(f'final_q {coh}: the iq LightGBM winner is not the five-family Q_SELECT winner')
    before = C.read_json(snap)
    parent_expected = dict(before['sha256_full_corpus_read_targets'])
    for key in ('sha256_incremental_q', 'sha256_track_a'):
        parent_expected.update({k: v for k, v in before[key].items() if k.endswith(('frozen_manifest.json', 'protocol.json')) or '/eval/predictions/MAIN_TEST' in k or '/models/' in k})
    parent_expected.update({k: v for k, v in before['sha256_cohort_role'].items() if '/cohorts/' in k})
    proto = dict(version=B.VERSION, role=B.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                 design='docs/CLAUDE_BALANCED_SHAP_20260916.md (Claude, continuing the Codex design line); no fitting in this run',
                 spec_sha256=C.sha256_file(B.SPEC), provenance_docs_sha256={p.name: C.sha256_file(p) for p in B.PROVENANCE_DOCS},
                 prior_test_exposure='Exploratory follow-up; the explained models were evaluated on MAIN TEST before (iq, Track A). Row selection uses hashes of keys only.',
                 parents=dict(read_only=['outputs/full_corpus_training_20260915', 'outputs/cohort_role_training_20260915', 'outputs/incremental_q_training_20260915',
                                         'outputs/track_a_mlp_20260916', 'source implementations', 'raw data', 'original manuscript'],
                              integrity_snapshot_before_sha256=C.sha256_file(snap), iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(B.IQ)),
                              ta_frozen_manifest_sha256=C.sha256_file(T.frozen_path(B.TA)), q_pre_only_schema_sha256=C.sha256_file(B.FC / 'q_pre_only_schema.json'),
                              cohort_manifest_sha256=C.sha256_file(B.CR / 'cohorts' / 'cohort_manifest.json'), expected_parent_file_sha256=parent_expected),
                 models=models, explained_quantity='final selected calibrated probability q (h90) of the frozen bundle, its own prediction definition',
                 groups=dict(record=gmap, sizes_by_order=[len(ix) for ix in gidx], source='parent q_pre_only_schema.json shap_groups restricted to the 352 ridge inputs',
                             note='unknown_objective_team_count stays in health_mana_other (parent classification)'),
                 rows=dict(explain_sets='MAIN_TEST h90-valid cohort rows; cells all and B40 (0.40 <= p_pre_V <= 0.60, frozen p_pre)',
                           n_explain=B.N_EXPLAIN, n_background=B.N_BG, n_cases=B.N_CASES,
                           explain_rule=f"first {B.N_EXPLAIN} by sha256('{B.TAG_EXPLAIN}:<cohort>:<cell>:<match>:<s>') ascending (model independent)",
                           background_rule=f"first {B.N_BG} cohort TRAIN h90-valid rows by sha256('{B.TAG_BG}:<cohort>:<match>:<s>'); same background for both cells and models",
                           case_rule=f"first {B.N_CASES} explained rows by sha256('{B.TAG_CASE}:<cohort>:<cell>:<match>:<s>')", overlap='rows in both cells reported'),
                 method=dict(reference='Lundberg & Lee (2017) arXiv:1705.07874; exact group Shapley by full enumeration of 2^7 = 128 coalitions (fc20260915_shap.group_exact_shapley)',
                             value_function='v(S) = mean over background rows of f(x_S, background_notS)',
                             bootstrap=dict(replicates=B.BOOT_REPS, seed=B.BOOT_SEED, unit='explained row', scope='explanation-sample variability only'),
                             bands=list(B.BANDS), threads='CPU; torch threads 1 for the MLP prediction definition'),
                 checks=['additivity phi.sum + base = f(x) < 1e-8', 'full coalition = f(x)', 'unique empty-coalition value', 'bitwise reload reproduction of the first 3 rows',
                         'deterministic selection', 'phi.sum + base equals the frozen parent evaluation prediction of the same rows (exact key join) < 1e-8',
                         'sealed access only after this run\'s freeze'],
                 not_claimed=['causal win factors', 'player skill', 'feature importance of unexplained models', 'cross-model or cross-cell magnitude rankings beyond description'],
                 smoke=dict(rows='TRAIN-only pseudo cells (all, and B40 by frozen p_pre) of the cohort', n_explain=B.SMOKE_EXPLAIN, n_background=B.SMOKE_BG),
                 environment=B.environment(), source_sha256=B.own_source_hashes())
    sha = C.write_json(B.OUT / 'protocol.json', proto)
    print('protocol sha256', sha)


if __name__ == '__main__':
    main()
