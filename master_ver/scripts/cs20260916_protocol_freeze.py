"""Stages P+Z: protocol.json (once) and the frozen manifest (plan freeze; no fitting). --smoke writes the smoke freeze under
smoke_train_only/. The full freeze requires passed contract tests, the Part B (Rashomon) results and a passed smoke run, and pins
their hashes; MAIN TEST rows for Part A are opened only after the full freeze exists."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402
import bsh20260916_common as B  # noqa: E402
import cs20260916_common as CS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    CS.log_command()
    base = CS.SMOKE if args.smoke else CS.OUT
    fz_fc = C.read_json(CS.FC / 'frozen_manifest.json')
    adapter = C.load_v_adapter(CS.V_ADAPTER, fz_fc['v_final_sha256'])
    schema = C.read_json(CS.FC / 'q_pre_only_schema.json')
    ridge, names = list(schema['predictor_sets']['ridge']), list(schema['input_names_all'])
    smap = CS.StateMap(adapter, names, ridge)
    gidx = smap.base_groups()
    if not (CS.OUT / 'protocol.json').exists():
        snap = CS.OUT / 'integrity' / 'snapshot_before.json'
        if not snap.exists():
            raise SystemExit('integrity snapshot_before.json must precede the protocol')
        before = C.read_json(snap)
        winners = {m: dict(zip(('bundle', 'calibration', 'bundle_sha256', 'candidate'), (str(Path(p).relative_to(CS.ROOT).as_posix()), cal, sha, cand)))
                   for m in CS.MODELS for (p, cal, sha, cand) in [CS.parent_winner(m, 'T')]}
        cands = {coh: [dict(source=s, family=f, config=cfg, bundle=str(p.relative_to(CS.ROOT).as_posix()), bundle_sha256=C.sha256_file(p)) for s, f, cfg, p in CS.candidate_bundles(coh)] for coh in CS.COHORTS}
        proto = dict(version=CS.VERSION, role=CS.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'), design='docs/CLAUDE_COMPOSITE_SHAP_20260916.md (Claude, continuing the Codex line); no fitting',
                     spec_sha256=C.sha256_file(CS.SPEC), provenance_docs_sha256={p.name: C.sha256_file(p) for p in CS.PROVENANCE_DOCS},
                     prior_test_exposure='Exploratory follow-up; the explained MAIN TEST rows were already used by the balanced-SHAP stage and evaluated before.',
                     parents=dict(read_only=['outputs/full_corpus_training_20260915', 'outputs/incremental_q_training_20260915', 'outputs/track_a_mlp_20260916', 'outputs/champion_class_20260916', 'outputs/balanced_shap_20260916'],
                                  integrity_snapshot_before_sha256=C.sha256_file(snap), adapter_sha256=fz_fc['v_final_sha256'], q_pre_only_schema_sha256=C.sha256_file(CS.FC / 'q_pre_only_schema.json'),
                                  bsh_frozen_manifest_sha256=C.sha256_file(B.frozen_path(CS.BSH)), bsh_shap_summary_sha256=C.sha256_file(CS.BSH / 'shap' / 'shap_summary.json'),
                                  iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(CS.IQ)), ta_frozen_manifest_sha256=C.sha256_file(T.frozen_path(CS.TA)),
                                  expected_parent_file_sha256=dict(before['sha256_full_corpus_read_targets'])),
                     part_a=dict(models=winners, cohorts=list(CS.SHAP_COHORTS), cells=list(CS.CELLS), n_explain=CS.N_EXPLAIN, n_background=CS.N_BG,
                                 row_selection='identical to balanced_shap_20260916 (same tags and contexts; equality checked)', base_columns=len(smap.base_names),
                                 derived_columns=len(smap.derived), derived_rule='<x>_x_time = <x> * time_minutes / 30; time_minutes_sq = time_minutes^2; snapshot_age_s = 0 (unused by V and q)',
                                 groups=list(CS.BASE_GROUPS), group_sizes={g: len(ix) for g, ix in zip(CS.BASE_GROUPS, gidx)}, group_indices={g: ix for g, ix in zip(CS.BASE_GROUPS, gidx)},
                                 base_names_sha256=C.sha256_json(smap.base_names),
                                 value_function='mean over background rows of F(masked base): derived columns recomputed, p_pre = V(reconstructed state), q = frozen calibrated bundle',
                                 explanation_type='interventional (marginal background) group Shapley of the composite F; not conditional, not causal'),
                     part_b=dict(rows='MAIN_VALIDATION Q_SELECT rows of the cohort (unsealed)', cohorts=list(CS.COHORTS), candidates=cands, calibrations=list(CS.CALS),
                                 loss='match-weighted Brier', epsilons=list(CS.EPSILONS), primary_epsilon=CS.PRIMARY_EPS, reliance='mean over 5 permutation seeds of Brier(group columns jointly permuted) - Brier(original)',
                                 permutation_seeds=list(CS.PERM_SEEDS), groups=list(CS.INPUT_GROUPS), group_sizes=B.EXPECTED_GROUP_SIZES),
                     checks=['reconstruction of stored inputs and p_pre from the base columns (tolerance recorded)', 'F(x) equals the parent stored MAIN TEST prediction (< 1e-9)',
                             'efficiency: sum phi + base = F(x) and v(full) = F(x) (< 1e-9)', 'explained/background rows equal the balanced-SHAP rows', 'rerun of first rows bitwise equal',
                             'Part B: candidate predictions equal the parents saved Q_SELECT candidate columns', 'sealed access only after the full freeze', 'parents unchanged'],
                     not_done=['conditional Shapley', 'path-specific attribution', 'N cohort for Part A', 'true model-class MCR optimisation', 'changes to models/labels/selections'],
                     smoke=dict(rows='TRAIN-only pseudo cells (train_fold 4 as pseudo-test, others as background)', n_explain=CS.SMOKE_EXPLAIN, n_background=CS.SMOKE_BG),
                     environment=CS.environment(), source_sha256=CS.own_source_hashes())
        print('protocol sha256', C.write_json(CS.OUT / 'protocol.json', proto))
    if CS.frozen_path(base).exists():
        raise SystemExit('frozen manifest exists (never overwritten)')
    access = [json.loads(l) for l in (base / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (base / 'access_log.jsonl').exists() else []
    if any(a.get('sealed') for a in access):
        raise SystemExit('sealed access before freeze')
    pinned = {}
    if not args.smoke:
        ct = CS.OUT / 'contract_tests' / 'result.json'
        if not ct.exists() or not C.read_json(ct).get('passed'):
            raise SystemExit('contract tests must pass before the full freeze')
        for coh in CS.COHORTS:
            p = CS.OUT / 'rashomon' / f'{coh}.json'
            if not p.exists():
                raise SystemExit(f'Part B results missing for {coh}')
            pinned[f'rashomon/{coh}.json'] = C.sha256_file(p)
        sm = CS.SMOKE / 'shap' / 'shap_summary.json'
        if not (CS.frozen_path(CS.SMOKE).exists() and sm.exists() and C.read_json(sm)['all_checks_pass']):
            raise SystemExit('smoke must complete and pass before the full freeze')
        pinned['contract_tests/result.json'] = C.sha256_file(ct)
        if (CS.OUT / 'shap').exists():
            raise SystemExit('shap/ exists before freeze')
    frozen = dict(role=CS.ROLE_TAG, version=CS.VERSION, smoke=bool(args.smoke), protocol_sha256=C.sha256_file(CS.OUT / 'protocol.json'), adapter_sha256=fz_fc['v_final_sha256'],
                  adapter_file_sha256_now=C.sha256_file(CS.V_ADAPTER), base_names_sha256=C.sha256_json(smap.base_names), group_sizes={g: len(ix) for g, ix in zip(CS.BASE_GROUPS, gidx)},
                  models={m: dict(bundle_sha256_now=C.sha256_file(CS.parent_winner(m, 'T')[0]), bundle_sha256_parent=CS.parent_winner(m, 'T')[2]) for m in CS.MODELS},
                  pinned_files_sha256=pinned, source_sha256=CS.own_source_hashes(), frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                  statement='Groups, rows, models and Part B results are frozen here; MAIN TEST rows are opened for Part A only after this file exists.')
    if frozen['adapter_file_sha256_now'] != frozen['adapter_sha256'] or any(v['bundle_sha256_now'] != v['bundle_sha256_parent'] for v in frozen['models'].values()):
        raise SystemExit('parent adapter / bundle hash differs from the parent freeze')
    print('frozen sha256', C.write_json(CS.frozen_path(base), frozen))


if __name__ == '__main__':
    main()
