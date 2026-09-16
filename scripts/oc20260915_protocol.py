"""Objective-channel ablation stage P: protocol.json, ordered feature evidence, analysis choices, parent inventory/hashes.

Written BEFORE any B_noobj fit (including TRAIN smoke fits). Fails if the frozen token rule does not yield exactly
361 primary / 176 dropped / 185 retained columns. Refuses to overwrite.
"""
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
import lv20260915_common as LV  # noqa: E402
import oc20260915_common as K  # noqa: E402

import platform  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402


def source_trace(n):
    """Which StateV2 code block (gameplay/state_value.py StateBuilder.at, renamed by state_value_v2) produces a column."""
    base = n[:-len('_x_time')] if n.endswith('_x_time') else n
    inter = ' x query_minutes/30 interaction' if n.endswith('_x_time') else ''
    m = re.match(r'participant_slot(\d)_(.+)$', n)
    if n in ('time_minutes', 'time_minutes_sq'):
        return 'query time'
    if n == 'snapshot_age_s':
        return 'observation age (excluded from A and B)'
    if n == K.UNKNOWN_COL:
        return 'mixed unknown-team event count (objective/soul/structure)'
    if m:
        f = m.group(2)
        if f in ('totalGold_norm', 'curGold_norm', 'level_norm', 'xp_norm', 'hp_pct', 'mp_pct', 'alive', 'laneCS_norm', 'jgCS_norm'):
            return 'participant frame snapshot (SNAPSHOT_FIELDS, latest frame <= query)'
        if f == 'champion_id':
            return 'participant champion id (categorical)'
        if f in ('kills', 'deaths', 'death_since_snapshot', 'death_last_30s', 'death_age_minutes'):
            return 'participant CHAMPION_KILL history'
        if f.endswith('_death_since_acquisition'):
            return 'participant death after team baron/elder acquisition (objective-derived buff proxy)'
    if re.match(r'(blue|red)_(kills)$', base):
        return 'team credited kills' + inter
    if re.search(r'_(tower_|plates|inhibitor_kills)', base):
        return 'team structure counts (BUILDING_KILL / TURRET_PLATE_DESTROYED)' + inter
    if re.search(r'_(dragons|dragon_)', base):
        return 'team elemental dragon counts (ELITE_MONSTER_KILL DRAGON)' + inter
    if '_soul_' in base:
        return 'team owned dragon soul (DRAGON_SOUL_GIVEN with team)' + inter
    if re.search(r'_(baron|elder|herald|horde|atakhan)', base):
        return 'team epic monster counts/history/age (ELITE_MONSTER_KILL)' + inter
    return 'UNTRACED'


def analysis_choices():
    return {
        'status_of_choices': 'fixed by the Codex specification; our design choices or inherited operational definitions, not prescribed by cited papers',
        'comparison': dict(
            A='untouched parent primary: expanded StateV2 logistic C=.01, raw final + 5 OOF adapters',
            B_noobj='same train.state_value_experiment.logistic (median impute + StandardScaler numeric, one-hot champions, liblinear, '
                    'random_state 7), C=.01, equal-match weights, TRAIN-only preprocessing inside the Pipeline, same folds; columns = '
                    'primary expanded columns minus EXACTLY those whose lower-cased name contains baron/elder/dragon/soul/herald/horde/atakhan',
            retained_note='kills/deaths/alive/HP/MP, economy/growth/CS/champions, time, towers/plates/inhibitors and their interactions, '
                          'and the mixed unknown_objective_team_count are kept verbatim; snapshot_age_s excluded as in A',
            scope='EXPLICIT NAMED OBJECTIVE-CHANNEL ABLATION; gold/XP/CS, structures, time and the mixed missing-team count can retain '
                  'indirect objective information; not a conditional causal effect and not objective-free data',
            max_iter=f'1500 as A; repeat the same fit with {list(K.MAX_ITER_LADDER)} only if n_iter_ >= max_iter; every attempt logged'),
        'fit_rows': 'exact parent TRAIN bucket query keys (424160 queries / 74168 matches); final on all TRAIN; fold-k on the other 4 folds',
        'calibration': dict(
            primary='RAW B probabilities (matching A raw); primary W, labels, agreement, objectives and q tables use raw B only',
            secondary='positive-slope sigmoid fitted on V_CAL for the final base and each fold base; raw vs sigmoid chosen on V_SELECT '
                      'by match-weighted log loss, then Brier, then name; chosen family frozen before TEST and reported separately',
            sign_invariance='shared strictly increasing calibration cannot change sign(delta) except clipping/finite precision; verified '
                            'on all sets as an implementation check, not independent label evidence'),
        'w_evaluation': dict(
            sets=['MAIN_TEST', 'EXT_KR_16.13', 'EXT_KR_16.14_pilot', 'EXT_KR_16.15', 'EXT_NA1_16.13', 'MAIN_V_CAL', 'MAIN_V_SELECT',
                  'MAIN_Q_CAL', 'MAIN_Q_SELECT', 'MAIN_TRAIN held-out fold'],
            keys='identical saved bucket query keys (is_bucket_sample = 1); A predictions must equal the parent saved values',
            metrics='AUC, Brier, log loss with equal-match weights (primary table) AND row weights; logistic recalibration slope/intercept, '
                    'CITL, 10-bin ECE; time bands <2 (explicit), 2-10, 10-20, 20-30, 30+ minutes',
            bootstrap=f'MAIN_TEST: {K.BOOT_REPS} paired match bootstrap (seed {K.BOOT_SEED}) of B-minus-A Brier and log loss under both '
                      'weightings; fixed-model uncertainty only'),
        'labels': dict(sets=list(K.LABEL_SETS), horizons=list(K.HS), primary_horizon=90,
                       adapter='TRAIN row -> B OOF adapter of its fold; other sets -> B final; the same adapter at q_pre and endpoint',
                       formula='delta = V_B(endpoint state) - V_B(q_pre state); Y = 1[delta > 0]; exact zeros counted',
                       identity='keys, endpoints, valid masks, snapshots equal to parent label files; A recomputed bit-for-bit'),
        'agreement': dict(metrics=['sign disagreement row / equal-match', 'positive rates', 'exact zeros', 'delta difference B-A mean, '
                                   'mean/median/p90 abs, Pearson, Spearman'],
                          bootstrap=f'h90 E/T/N for every main role and external set: {K.BOOT_REPS} paired match bootstrap, seed {K.BOOT_SEED}',
                          indicators='rows, matches, empty, sparse (<30 matches)'),
        'strata_h90': dict(abs_delta_A=['[0,.005]', '(.005,.01]', '(.01,.02]', '>.02'], start_minutes=['<2', '[2,10)', '[10,20)', '[20,30)', '>=30'],
                           p_pre_A=['[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,1]'], ending_reason='parent tied reason string',
                           post_snapshot_after_L=['1', '0'], same_pre_post_frame=['1 (post frame == pre frame)', '0'],
                           note='diagnostic only; no exclusion or threshold change'),
        'objectives_h90': dict(
            source='parent label counters during_counts (q_pre, L] and after_counts_h90 (L, e]; element team from lv20260915 observed '
                   'StateV2 differences over (q_pre, e] with exact (match, s) key verification; no new raw decoding',
            windows=['full (q_pre, e]', 'after_last_kill (L, e]'],
            categories=['baron', 'dragon (all elemental, team)', 'dragon_AIR/EARTH/FIRE/WATER/HEXTECH/CHEMTECH/OTHER', 'elder', 'herald',
                        'horde (grubs)', 'atakhan', 'soul_owned'],
            subgroups=dict(descriptive=['blue_only', 'red_only'],
                           diagnostic=['both_teams', 'multiple_acquisitions (count >= 2)', 'unknown_team_present', 'soul_teamid0_unassigned',
                                       'element team ambiguous in (L, e] when the element was also acquired in (q_pre, L]']),
            element_team_rule='(q_pre, e]: StateV2 element count difference per team; (L, e]: the same team difference only when that '
                              'element had no acquisition in (q_pre, L] (all interval acquisitions after L), otherwise ambiguous',
            statistics=['rows, matches, sparse', 'A-B disagreement row / equal-match', 'mean delta A and B', 'small |delta|<=.01 A and B',
                        'P(Y=1) A and B', 'acquisition-team-oriented delta (+delta for Blue-only, -delta for Red-only) for A, B and A-B'],
            reference='rows without a known-team acquisition in the window (pre-existing objective history possible)',
            interpretation='descriptive; not event causal value, not timing counterfactual, not immediate object reward; overlapping; '
                           'sparse and pilot cells carry no broad conclusion'),
        'q_dependence': dict(predictions='cohort_role_training_20260915 A_MAIN_TEST_h90_{T,N}.npz chosen specialists (T spec_ridge_raw, '
                                         'N spec_ridge_isotonic)', keys='exact main TEST h90 (match, s)', reproduction='A scores equal '
                                         'tog_readiness_audit calculations.json', scope='target dependence; no refit, no ranking'),
        'unknown_objective_team_count': dict(semantics=K.UNKNOWN_SEMANTICS,
                                             report='nonzero prevalence (row and equal-match) on TRAIN and every evaluation bucket set and on '
                                                    'engagement pre states of every label set'),
        'multiplicity': 'no adjustment; intervals descriptive',
    }


def main():
    st = K.Status('protocol')
    K.log_command()
    try:
        if (K.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json exists; refusing to overwrite')
        if (K.OUT / 'models').exists() or (K.OUT / 'smoke_train_only').exists():
            raise SystemExit('fitted artifacts exist before the protocol')
        man = C.read_json(K.FC / 'extract' / 'MAIN' / 'extraction_manifest.json')
        names = man['names']
        vman = C.read_json(K.FC / 'v_models_manifest.json')
        primary = vman['feature_names']
        if primary != LV.expanded_features(names):
            raise SystemExit('primary A features differ from expanded rule')
        retained, dropped = K.feature_split(primary)
        counts = dict(primary=len(primary), dropped=len(dropped), retained=len(retained))
        if counts != K.EXPECTED:
            raise SystemExit(f'feature count discrepancy {counts} != {K.EXPECTED}; resolve source before fitting')
        untraced = [n for n in primary if source_trace(n) == 'UNTRACED']
        if untraced:
            raise SystemExit(f'untraced columns {untraced[:5]}')
        by_trace = {}
        for n in primary:
            key = ('DROPPED: ' if n in dropped else 'RETAINED: ') + source_trace(n)
            by_trace[key] = by_trace.get(key, 0) + 1
        ev = dict(rule='drop a primary expanded column iff its lower-cased name contains any of ' + ', '.join(K.OBJECTIVE_TOKENS),
                  tokens=list(K.OBJECTIVE_TOKENS), counts=counts, state_names_sha256=C.sha256_json(names),
                  primary_ordered=primary, primary_sha256=C.sha256_json(primary),
                  retained_ordered=retained, retained_sha256=C.sha256_json(retained),
                  dropped_ordered=[dict(name=n, tokens=K.dropped_token(n), source=source_trace(n)) for n in dropped],
                  dropped_sha256=C.sha256_json(dropped), retained_sources={n: source_trace(n) for n in retained},
                  columns_by_source=by_trace, snapshot_age_s='not in A and not in B (audit metadata only)',
                  unknown_objective_team_count=dict(retained=K.UNKNOWN_COL in retained, semantics=K.UNKNOWN_SEMANTICS),
                  source_code=dict(state_value=str(C.WT / 'gameplay' / 'state_value.py'), state_value_sha256=C.sha256_file(C.WT / 'gameplay' / 'state_value.py'),
                                   state_value_v2=str(C.WT / 'gameplay' / 'state_value_v2.py'), state_value_v2_sha256=C.sha256_file(C.WT / 'gameplay' / 'state_value_v2.py'),
                                   primary_features_source='outputs/full_corpus_training_20260915/v_models_manifest.json feature_names'))
        ev_sha = C.write_json(K.OUT / 'feature_evidence.json', ev)
        st.update('running', 'parent_inventory', counts=counts, next_step='hashes')
        inv = LV.parent_file_inventory(K.parent_roots())
        inv_sha = C.write_json(K.OUT / 'integrity' / 'parent_inventory_before.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                                         roots=[str(r) for r in K.parent_roots()], files=len(inv), inventory=inv))
        targets = K.read_parent_hash_targets()
        hashes = {str(p.relative_to(K.ROOT)): C.sha256_file(p) for p in targets}
        par_sha = C.write_json(K.OUT / 'integrity' / 'parent_hashes_before.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), files=len(hashes), sha256=hashes))
        src_sha = C.write_json(K.OUT / 'source_hashes.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                   sources={str(p): C.sha256_file(p) for p in K.source_files() if p.exists()}))
        import numpy, scipy, sklearn, joblib
        protocol = dict(
            version=K.VERSION, role=K.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            design='docs/CLAUDE_OBJECTIVE_CHANNEL_ABLATION_20260915.md (Codex, frozen); implementation and execution Claude Opus 5',
            spec_sha256=C.sha256_file(K.SPEC), findings_doc_sha256=C.sha256_file(K.FINDINGS_DOC), audit_doc_sha256=C.sha256_file(K.AUDIT_DOC),
            written_before=['any B_noobj fit (including TRAIN smoke fits)', 'any TEST/external/Q_CAL/Q_SELECT outcome access',
                            'any new TEST/external comparison'],
            prior_exposure=('EXPLORATORY, NOT CONFIRMATORY. Already seen before this protocol: parent full-corpus V/q TEST and external '
                            'results, cohort/role results, ToG audit recalculations, and the label-validity study (A vs B_reg/B_econ '
                            'agreement on TEST/external, observed-direction and objective tables, PH1 stale-frame mechanism).'),
            human_review='not part of this task (user scope decision); semantic validation remains unperformed',
            resources=dict(python=K.PYTHON, platform=platform.platform(), numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                           joblib=joblib.__version__, cpu='<= 4 concurrent single-threaded processes', gpu='not used',
                           prohibited=['installs', 'deletions', 'commits/pushes', 'credentials', 'external messages', 'writes to parents, raw caches, '
                                       'original repo, manuscript', 'GPU queue changes']),
            inputs=dict(parent_full=str(K.FC), parent_cohort=str(K.CR), parent_label_validity=str(K.LVO),
                        parent_v_models_manifest_sha256=C.sha256_file(K.FC / 'v_models_manifest.json'),
                        parent_frozen_manifest_sha256=C.sha256_file(K.FC / 'frozen_manifest.json'),
                        label_validity_frozen_manifest_sha256=C.sha256_file(K.LVO / 'frozen_manifest.json'),
                        feature_evidence_sha256=ev_sha, parent_hashes_before_sha256=par_sha, parent_inventory_before_sha256=inv_sha,
                        source_hashes_sha256=src_sha),
            parent_access='lv20260915_common.ParentReadOnly(base=this root): parent pure readers, preserved role checks, access logs here only',
            analysis=analysis_choices(),
            contracts=['exact feature removal (361/176/185, token rule, order, no token in retained)', 'retained values identical to A inputs',
                       'TRAIN key identity with parent (424160/74168) and fold hash', 'OOF own-match exclusion', 'preprocessing membership',
                       'same adapter at both label ends', 'freeze before sealed access', 'parents unchanged', 'exact label keys/endpoints/masks',
                       'A bit-for-bit', 'exact zeros and label formula', 'calibration sign invariance', 'observed-array key identity',
                       'independent metric/bootstrap recomputation', 'q A-score reproduction'],
            literature=dict(Maymin2021='doi:10.1515/jqas-2019-0096: state win probability change valuation rationale',
                            KimCoG2020='https://ieee-cog.org/2020/papers/paper_221.pdf: win-probability quality/calibration motivation, not reproduced',
                            JacobsWallach='arXiv:1912.05511: construct vs operationalized measurement',
                            not_prescribed='the 176-column rule, 90 s, C=.01, strata and thresholds are our choices or inherited operational definitions'),
            claims_not_made=['causal objective effect', 'objective-free data or complete removal of objective proxies', 'semantic correctness of labels',
                             'fresh confirmation', 'best q performance for B labels', 'human validation'],
        )
        protocol['protocol_sha256_of_content'] = C.sha256_json(protocol)
        sha = C.write_json(K.OUT / 'protocol.json', protocol)
        st.update('complete', 'protocol', protocol_sha256=sha, counts=counts, parent_files_hashed=len(hashes), parent_inventory_files=len(inv),
                  next_step='contracts')
        return 0
    except SystemExit as exc:
        st.update('failed', 'protocol', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'protocol', error=repr(exc), next_step='fix and rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
