"""Label validity stage P: protocol.json, exact feature lists, analysis choices and parent/source hash inventory.

Written BEFORE any new fit (including TRAIN-only smoke fits) and before any newly generated TEST/external
sensitivity score exists. Refuses to overwrite an existing protocol.json.
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
import lv20260915_common as K  # noqa: E402

import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402


def analysis_choices():
    return {
        'status_of_choices': 'prespecified diagnostic choices of this study (Codex design); not established by the cited papers',
        'A_primary': dict(
            identity='parent final V (models/v/v_final_raw.joblib) and five OOF fold adapters, untouched; primary labels, '
                     'E/T/N rules, cutoff q_pre = s-1 ms, h90 primary and h60/h120 sensitivity unchanged',
            verification='recompute A p_pre/p_post on every engagement row with the parent adapters and require exact '
                         'equality with the saved parent label arrays (differences recorded, saved values kept)'),
        'alternative_V': dict(
            models=K.MODEL_SPECS,
            estimator='train.state_value_experiment.logistic(cols, C): median impute + StandardScaler on numeric columns, '
                      'OneHotEncoder(handle_unknown=ignore) on *champion_id, LogisticRegression(liblinear, random_state=7)',
            max_iter='1500 as primary; if n_iter_ >= max_iter the SAME fit is repeated with max_iter from ladder '
                     f'{list(K.ALT_MAX_ITER_LADDER)} (convergence only; C and features never tuned)',
            fit_rows='exact parent TRAIN bucket query keys (expected 424160 rows / 74168 matches), equal total weight per match',
            preprocessing_membership='all preprocessing is inside the sklearn Pipeline and fitted only on the rows of that fit',
            folds='existing parent TRAIN folds (sub_role foldk from sha256(full-v-oof-20260915:+match)%5); fold-k adapter '
                  'fitted on TRAIN rows of the other four folds only',
            calibration_candidates='raw identity and positive-slope sigmoid fitted on V_CAL (parent PositiveSlopeSigmoid)',
            selection='V_SELECT match-weighted log loss, then Brier, then name (parent order); chosen family applied to every '
                      'fold adapter with fold-specific V_CAL calibration if sigmoid is chosen',
            prohibited=['grid search', 'C or feature tuning', 'q-label-based V selection', 'TEST/external W before freeze'],
            freeze='frozen_manifest.json with hashes of both comparator families, calibrators, selections and this protocol '
                   'is written before any TEST/external/Q_CAL/Q_SELECT W or any TEST/external generated label is read'),
        'v_evaluation': dict(
            sets=['MAIN_TEST (primary context)', 'EXT_KR_16.13', 'EXT_KR_16.14_pilot', 'EXT_KR_16.15', 'EXT_NA1_16.13',
                  'MAIN_V_CAL/V_SELECT (used in selection, diagnostic)', 'MAIN_Q_CAL/Q_SELECT (diagnostic)',
                  'MAIN_TRAIN held-out fold (diagnostic)'],
            rows='saved bucket query keys only (is_bucket_sample = 1)',
            metrics='match-weighted AUC, Brier, log loss, logistic recalibration slope/intercept, CITL, 10-bin ECE and bins '
                    '(parent C.evaluate)',
            time_bands_minutes=['<2 (explicit, expected empty: grid starts at 2 min)', '2-10', '10-20', '20-30', '30+'],
            uncertainty='MAIN_TEST only: 1000 paired match bootstrap of Brier/log-loss differences A-B (fixed models; '
                        'no AUC replicate CI)',
            interpretation='context for label comparisons: disagreement with a poorer V is not evidence that A is wrong'),
        'alternative_labels': dict(
            sets=list(K.LABEL_SETS), horizons=list(K.HS), keys='(match, s) in parent label row order; exact identity required',
            adapter='TRAIN row -> comparator OOF adapter of the row fold (sub_role); VALIDATION/TEST/external -> comparator final '
                    'adapter; the same adapter at q_pre and at the endpoint',
            formula='delta_h = V_B(state at endpoint_h) - V_B(state at q_pre); Y_h = 1[delta_h > 0]; exact zero -> 0, counted',
            validity='parent valid_h masks unchanged; invalid rows kept with NaN/-1 and their parent reasons',
            output='alt_labels/<set>_alt_labels.npz with source/fold/model metadata and adapter hashes'),
        'agreement': dict(
            pairs=[['A', 'B_reg'], ['A', 'B_econ']], cohorts=['E (all valid)', 'T (cohort code 1)', 'N (cohort code 0)'],
            sets='each main role (TRAIN OOF, VALIDATION, TEST) and each external set; all three horizons',
            metrics=['row-weighted disagreement', 'equal-match-weighted disagreement', 'positive rate per model',
                     'exact-zero delta count per model', 'delta difference: mean(B-A), mean/median/p90 |B-A|, Pearson, Spearman'],
            unchanged_endpoint='A-vs-B disagreement reported separately for rows whose endpoints are identical across 60/90/120 '
                               'and for rows whose endpoints differ; per-model horizon sign disagreement reported on all rows and '
                               'on changed-endpoint rows (unchanged-endpoint horizon disagreement must be exactly 0)',
            bootstrap=f'h90 E/T/N for every set: {K.BOOT_REPS} paired match bootstrap (seed {K.BOOT_SEED}); same match '
                      'multiplicities for both pairs and both weightings; percentile 95% intervals; fixed models, no '
                      'training uncertainty; no binomial row-independence intervals',
            indicators='each cell has rows, matches, empty flag, sparse flag (<30 matches)'),
        'strata_h90': dict(
            abs_delta_A=['[0,.005]', '(.005,.01]', '(.01,.02]', '>.02'],
            p_pre_A=['[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,1]'],
            start_minutes=['<2', '[2,10)', '[10,20)', '[20,30)', '>=30'],
            ending_reason='parent tied reason string (e.g. next_kill, horizon, next_kill|next_engagement_start)',
            post_snapshot_after_L=['1', '0'],
            post_frame_age_s=['[0,15)', '[15,30)', '[30,45)', '[45,60)', '>=60'],
            pre_frame_age_s=['[0,15)', '[15,30)', '[30,45)', '[45,60)', '>=60'],
            fine_scale=['pick (n_min<=1)', 'skirmish (2-3)', 'teamfight (>=4)', 'unknown'],
            note='diagnostic strata only: not label exclusion thresholds and not error bars for V'),
        'observed_outcomes': dict(
            intervals={'during': '(q_pre, L]  with q_pre = s - 1 ms and L = last kill of the engagement',
                       'after_last_kill': '(L, e_h]',
                       'full': '(q_pre, e_h] = during + after_last_kill'},
            kills='credited kills per team from StateV2 event counts (CHAMPION_KILL with killerId on that roster, '
                  'timestamp <= query): Blue kills - Red kills over during and full intervals; the endpoint rule makes '
                  '(L, e] kill-free, so both intervals are reported and their identity is verified per row; ties are a '
                  'separate category; uncredited kills (raw CHAMPION_KILL minus credited kills) are counted',
            resources='team difference (Blue sum - Red sum over five participant slots) of totalGold_norm and xp_norm at '
                      'the latest frame <= e minus the latest frame <= q_pre; NORMALIZED cache units (source normalizers '
                      'traced: gameplay/pipeline_cache.py DEN_TOT_G = 25000, DEN_XP = 20000, clip [0,5], float32); '
                      'post frame == pre frame -> stale_same_frame category (change not observed, not a tie)',
            objectives='parent raw event counters during/after (engagement_labels_v3_rules.count_events, lo exclusive, hi '
                       'inclusive) for baron, elder, herald, horde (grubs), atakhan, owned soul, dragon (team) and each '
                       'dragon element (no team in counters); per-element team from StateV2 differences over the full '
                       'interval only; unknown-team and teamId-0 soul are diagnostic only; subgroups overlap and are never '
                       'summed as disjoint',
            reporting='counts, sign tables, P(Y=1 | category) row and match weighted, small |delta_A|<=.01 proportion, '
                      'A/B disagreement per category',
            prohibited=['fixed gold exchange weights', 'causal objective uplift', "semantic 'accuracy' with heuristics as truth"]),
        'q_label_dependence': dict(
            predictions=['cohort_role_training_20260915/eval/predictions/A_MAIN_TEST_h90_{T,N}.npz (pooled and every '
                         'specialist candidate)', 'full_corpus_training_20260915/eval/predictions/q_MAIN_TEST_h90.npz (E pooled '
                         'candidates)'],
            keys='exact h90 main TEST (match, s) keys; primary Y must equal saved y',
            reproduction='primary scores must reproduce tog_readiness_audit calculations.json (T/N) and results_q.json (E)',
            scope='diagnostic label dependence of models trained on A labels; no ranking, selection or ceiling claim'),
        'review_packet': dict(
            source='MAIN_TRAIN h90 valid rows only (OOF labels); no VALIDATION/TEST/external case',
            strata=[dict(name='S1_T_small_abs_delta_A', k=40, rule='cohort T and |delta_A h90| <= 0.01'),
                    dict(name='S2_T_A_disagrees_with_either_B', k=40, rule='cohort T and (Y_A != Y_B_reg or Y_A != Y_B_econ)'),
                    dict(name='S3_T_observed_direction_conflict_with_objective', k=20,
                         rule='cohort T and an objective (baron/dragon/elder/herald/horde/atakhan/owned soul) acquired by a '
                              'known team in (q_pre, e] and (sign(kill diff full) or sign(gold diff change, non-stale frames '
                              'only) is nonzero and opposite to the A direction (+1 if Y_A = 1 else -1))'),
                    dict(name='S4_N_reference', k=20, rule='cohort N (any valid h90 row)')],
            priority='S1 > S2 > S3 > S4; at most one case per match in the whole packet (matches used by a higher '
                     'stratum are removed before lower strata)',
            ranking=f"match rank sha256('lv20260915-packet:{K.PACKET_SEED}:<stratum>:match:<match>'), within-match row rank "
                    f"sha256('lv20260915-packet:{K.PACKET_SEED}:<stratum>:row:<match>:<s>'); lowest ranks chosen",
            inclusion_probability='treating the hash as a uniform random permutation: P(case) = min(1, k / M_remaining) x '
                                  '1 / r_match, conditional on higher-priority selections (M_remaining = eligible matches not '
                                  'used earlier; r_match = eligible rows of the match in that stratum); shortages reported',
            case_order=f"reviewer order by sha256('lv20260915-case-order:{K.PACKET_SEED}:<match>:<s>') (hides stratum)",
            reviewer_view_hidden=['A/B probabilities', 'delta', 'generated Y', 'selection stratum', 'q', 'W / winner',
                                  'match id', 'player/account identifiers'],
            private_files=['PRIVATE_review_case_key.json (case -> match mapping)',
                           'PRIVATE_review_analysis_key.csv (A/B probabilities, delta, Y, stratum, q, W)'],
            form=['(a) observed short-term exchange: Blue / Red / neutral / insufficient',
                  '(b) strategic advantage over the shown interval: Blue / Red / neutral / insufficient',
                  '(c) endpoint: truncates a continuing fight / new encounter / unclear',
                  '(d) what observation is missing (free text)', '(e) confidence ordinal 1-5'],
            human_review='UNPERFORMED; reviewer names and ratings left empty; never filled with AI judgments'),
        'multiplicity': 'no multiple-comparison adjustment; all intervals are descriptive',
    }


def contracts():
    return ['data roles and freeze gate (pre-freeze outcome access limited to TRAIN folds, V_CAL, V_SELECT)',
            'OOF own-match exclusion for every TRAIN engagement (adapter fit set reconstructed and hashed)',
            'preprocessing membership (scaler n_samples_seen_ = fit rows; one-hot categories from fit rows)',
            'same adapter at both label ends', 'no forbidden future/objective fields in B_econ; feature lists exact',
            'no mutation of frozen parent artifacts (sha256 of read files + size/mtime inventory of both parent roots)',
            'exact cohort joins (match, s, L) and unchanged valid masks',
            'zero-delta and kill-tie logic (synthetic)', 'event interval counting against raw TRAIN cache fixtures',
            'normalization source (gold_team_minute vs totalGold_norm x 25000 on TRAIN fixtures; integer-grid checks)',
            'reload predictions (npz reload; re-predict with reloaded adapters)',
            'independently recomputable metrics and bootstrap', 'reviewer blind-field absence']


def main():
    st = K.Status('protocol')
    K.log_command()
    try:
        if (K.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json already exists; refusing to overwrite')
        existing_fit = [p for p in (K.OUT / 'models').rglob('*.joblib')] if (K.OUT / 'models').exists() else []
        if existing_fit or (K.OUT / 'smoke_train_only').exists():
            raise SystemExit('fitted artifacts exist before the protocol')
        st.update('running', 'feature_lists', next_step='hash inventory')
        man = C.read_json(K.FC / 'extract' / 'MAIN' / 'extraction_manifest.json')
        names = man['names']
        vman = C.read_json(K.FC / 'v_models_manifest.json')
        a_feats = vman['feature_names']
        if a_feats != K.expanded_features(names):
            raise SystemExit('primary A feature list differs from the expanded rule')
        feats = dict(state_names=names, state_names_sha256=C.sha256_json(names), state_version=C.STATE_VERSION,
                     A_primary=dict(features=a_feats, n=len(a_feats), C=C.V_C, source='parent v_models_manifest.json'),
                     B_reg=dict(features=K.model_features('B_reg', names), C=K.MODEL_SPECS['B_reg']['C']),
                     B_econ=dict(features=K.model_features('B_econ', names), C=K.MODEL_SPECS['B_econ']['C'],
                                 numeric=[n for n in K.econ_features(names) if not n.endswith('champion_id')],
                                 categorical=[n for n in K.econ_features(names) if n.endswith('champion_id')]))
        for m in K.MODELS:
            feats[m]['n'] = len(feats[m]['features'])
            feats[m]['sha256'] = C.sha256_json(feats[m]['features'])
        feat_sha = C.write_json(K.OUT / 'feature_lists.json', feats)

        st.update('running', 'parent_inventory', next_step='sha256 of read parent files')
        inv = K.parent_file_inventory([K.FC, K.CR])
        inv_sha = C.write_json(K.OUT / 'integrity' / 'parent_inventory_before.json',
                               dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), files=len(inv), inventory=inv))
        targets = K.read_parent_hash_targets()
        hashes = {}
        for i, p in enumerate(targets):
            hashes[str(p.relative_to(K.ROOT))] = C.sha256_file(p)
            if i % 200 == 0:
                st.update('running', 'parent_hashes', processed=i, total=len(targets))
        par_sha = C.write_json(K.OUT / 'integrity' / 'parent_hashes_before.json',
                               dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), files=len(hashes), sha256=hashes))
        src = {str(p): C.sha256_file(p) for p in K.source_files()}
        src_sha = C.write_json(K.OUT / 'source_hashes.json', dict(written_at=time.strftime('%Y-%m-%d %H:%M:%S'), sources=src))

        import numpy, scipy, sklearn, joblib
        protocol = dict(
            version=K.VERSION, role=K.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            design='docs/CLAUDE_LABEL_VALIDITY_FULL_20260915.md (Codex); implementation and execution Claude Opus 5',
            spec_sha256=C.sha256_file(K.SPEC), audit_doc_sha256=C.sha256_file(K.AUDIT_DOC),
            written_before=['any comparator fit (including TRAIN-only smoke fits)', 'any TEST/external/Q_CAL/Q_SELECT W access',
                            'any generated TEST/external comparator label or sensitivity score'],
            prior_exposure=('EXPLORATORY FOLLOW-UP, NOT A NEW CONFIRMATORY TEST. Existing results were already seen before this '
                            'protocol: parent full-corpus V and q TEST/external metrics (REPORT.md, results_v/q.json), cohort/role '
                            'TEST/external results, and the ToG readiness audit recalculations (primary TEST h90 label '
                            'distributions, abs(delta) fractions, horizon sign flips, T/N metrics). Primary labels on TEST and '
                            'external sets were generated and inspected by earlier stages.'),
            resources=dict(python=K.PYTHON, platform=platform.platform(), numpy=numpy.__version__, scipy=scipy.__version__,
                           sklearn=sklearn.__version__, joblib=joblib.__version__,
                           cpu='<= 4 concurrent single-threaded processes (BLAS/OMP threads = 1)', gpu='not used',
                           prohibited=['installs', 'deletions', 'commits/pushes', 'credentials', 'external messages',
                                       'writes to parents, raw D: caches/shards, the original Pycharm repo or manuscript',
                                       'GPU queue changes']),
            inputs=dict(parent_full=str(K.FC), parent_cohort=str(K.CR), main_cache=str(C.CACHE_MAIN),
                        parent_frozen_manifest_sha256=C.sha256_file(K.FC / 'frozen_manifest.json'),
                        parent_v_models_manifest_sha256=C.sha256_file(K.FC / 'v_models_manifest.json'),
                        cohort_frozen_manifest_sha256=C.sha256_file(K.CR / 'frozen_manifest.json'),
                        feature_lists_sha256=feat_sha, parent_hashes_before_sha256=par_sha,
                        parent_inventory_before_sha256=inv_sha, source_hashes_sha256=src_sha,
                        raw_access='raw cache files only for review-packet matches (TRAIN) and TRAIN sanity fixtures; '
                                   'external raw timelines only for post-freeze normalization fixtures'),
            parent_access='ParentReadOnly wrapper: parent pure readers; outcome/label access checks preserved and logged to '
                          'this root (outcome_access_log.jsonl, label_access_log.jsonl); parent logs must stay unchanged',
            analysis=analysis_choices(), contracts=contracts(),
            literature=dict(
                Maymin2021='DOI 10.1515/jqas-2019-0096: state win probability and win-probability change as valuation rationale',
                KimCoG2020='CoG 2020 LoL win probability with calibration emphasis: probability-quality motivation, not reproduced',
                JacobsWallach2021='Measurement and Fairness (FAccT 2021, arXiv:1912.05511): construct vs operationalized measurement '
                                  'and construct validity are distinct',
                event_boundary_CIF='Austin, Lee & Fine 2016 competing-risks cumulative incidence and the project event-boundary '
                                   'CIF results support methodological handling of competing endpoints, NOT 90 s or true fight '
                                   'attribution',
                our_choices='the two comparator feature/C settings, strata, bins and the 120-case packet are prespecified diagnostic '
                            'choices of this study; papers do not establish them'),
            claims_not_made=['construct validity of labels from numeric agreement', 'semantic accuracy of Y',
                             'causal objective or kill value', 'that B is a better or truer V', 'new q validation or ranking',
                             'expert review performed'],
        )
        protocol['protocol_sha256_of_content'] = C.sha256_json(protocol)
        sha = C.write_json(K.OUT / 'protocol.json', protocol)
        st.update('complete', 'protocol', protocol_sha256=sha, parent_files_hashed=len(hashes), parent_inventory_files=len(inv),
                  next_step='contract tests (synthetic + TRAIN-only)')
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
