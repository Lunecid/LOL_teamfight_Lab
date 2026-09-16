"""Write outputs/full_corpus_training_20260915/protocol.json BEFORE any extraction or fitting.

The protocol freezes the authorized design of docs/CLAUDE_FULL_CORPUS_TRAIN_20260915.md with exact formulas,
input hashes and resource limits. It refuses to overwrite an existing protocol (amendments go to
protocol_amendments.json with timestamps, and must precede the stage they affect).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402

GOVERNING = ['docs/CLAUDE_FULL_CORPUS_TRAIN_20260915.md', 'docs/FULL_CORPUS_SPLIT_CORRECTION_20260915.md',
             'docs/EXTERNAL_TEST_PLAN_20260915.md', 'docs/CLAUDE_EXECUTE_P4_Q_BASELINES.md',
             'docs/CLAUDE_EXECUTE_P3_LABEL_SENSITIVITY.md', 'docs/CLAUDE_EXECUTE_P2_TEMPORAL_V3.md',
             'docs/CLAUDE_EXECUTE_P1_V2.md', 'docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md',
             'outputs/full_corpus_preflight_20260915/REPORT.md', 'outputs/full_corpus_preflight_20260915/protocol.json',
             'outputs/full_corpus_preflight_20260915/validation.json', 'outputs/full_corpus_preflight_20260915/status.json']
INPUTS = ['outputs/full_corpus_preflight_20260915/main_matches.csv', 'outputs/postkill_objective_delay_full/exposures.csv',
          'outputs/postkill_objective_delay_full/run.json'] + \
         [f'outputs/full_corpus_preflight_20260915/external/external_{s}.csv' for s in
          ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13', 'EUW1_complete')]
WT_SOURCES = ['gameplay/state_value.py', 'gameplay/state_value_v2.py', 'train/temporal_winprob.py',
              'train/state_value_experiment.py', 'train/independent_winprob_v2.py', 'train/temporal_history_winprob_v3.py',
              'data/cache_io.py', 'core/config.py', 'core/presets.py', 'gameplay/fights.py', 'data/index_split.py',
              'gameplay/pipeline_cache.py', 'gameplay/pipeline.py']
ROOT_SOURCES = ['scripts/engagement_labels_v3_rules.py', 'scripts/measure_postkill_objective_delay.py',
                'scripts/measure_postkill_full.py', 'scripts/run_q_v3_baselines.py', 'scripts/train_independent_v2.py',
                'scripts/run_engagement_labels_v3_sensitivity.py', 'scripts/run_temporal_winprob_v3.py',
                'scripts/fc20260915_common.py', 'scripts/fc20260915_extract.py', 'scripts/fc20260915_protocol.py']


def h(rel):
    p = C.ROOT / rel
    return C.sha256_file(p) if p.exists() else None


def main():
    path = C.OUT / 'protocol.json'
    if path.exists():
        raise SystemExit('protocol.json exists; write amendments to protocol_amendments.json instead')
    run = C.read_json(C.ROOT / 'outputs/postkill_objective_delay_full/run.json')
    detector_sources = {}
    for p, recorded in run['source_hashes'].items():
        cur = hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
        detector_sources[str(Path(p).relative_to(C.REPO)).replace('\\', '/')] = dict(parent_run=recorded, current=cur,
                                                                                        equal=cur == recorded)
    protocol = dict(
        version=C.VERSION, role=C.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        written_before=['any cache decoding of this run', 'any model fitting', 'any label generation', 'any TEST scoring'],
        design='Codex specification docs/CLAUDE_FULL_CORPUS_TRAIN_20260915.md; Claude Opus 5 implementation and execution',
        authorization='User explicitly requested full-data retraining on 2026-09-15; supersedes the overnight deadline and '
                      'the preflight-only restriction.',
        governing_sha256={g: h(g) for g in GOVERNING},
        inputs_sha256={i: h(i) for i in INPUTS},
        sources_sha256=dict(worktree={s: h('worktrees/engagement-state-value/' + s) for s in WT_SOURCES},
                            root={s: h(s) for s in ROOT_SOURCES},
                            original_repo_detector_vs_parent_run=detector_sources),
        environment=dict(python='C:/Users/todtj/anaconda3/python.exe', cpu_limit='<= 4 concurrent worker threads/processes in total',
                         blas='OMP/MKL/OPENBLAS/NUMEXPR threads = 1 in every process (no nesting); LightGBM n_jobs <= 4 alone',
                         gpu='not used (CUDA_VISIBLE_DEVICES empty); user GPU queue untouched',
                         prohibited=['installs', 'deletion', 'commits/pushes', 'secrets', 'writes to raw D: inputs, cache, collector DBs',
                                     'modifying old models/source/artifacts or the original Pycharm repository']),
        corpus=dict(
            main=dict(manifest='outputs/full_corpus_preflight_20260915/main_matches.csv (authoritative)',
                      roles={k: dict(patch=v[0], matches=v[1]) for k, v in C.MAIN_EXPECTED.items()}, total=C.MAIN_TOTAL,
                      rule='every eligible match contributes to its designated role; no subsampling; games without engagements '
                           'still belong to the V dataset; every exclusion reported with match id and reason'),
            exposures=dict(source='outputs/postkill_objective_delay_full/exposures.csv (parent full-corpus detector run, v3.3)',
                           rows=566452, matches=194676,
                           rule='all parent exposure rows enter boundary reconstruction; overlap next_start<=L and any other invalid '
                                'support are excluded with lineage; eligibility is decided by the B-rule validity flags per horizon'),
            unknown_team_1417=('Parent run.json stats.unknown_team counts ELITE_MONSTER_KILL events whose killer team is not '
                               '100/200 inside measure_postkill_objective_delay.analyse_match; that function only skips those '
                               'events from objective-delay pairs. Exposure rows (s, L, next_start, end) are computed from '
                               'detector refs and GAME_END and are unaffected, so it is not an engagement eligibility rule. '
                               'In StateV2 such events increment unknown_objective_team_count (a pre-state feature). The '
                               'extraction census re-counts them per role.'),
            external=dict(manifests='outputs/full_corpus_preflight_20260915/external/*.csv',
                          sets={k: dict(expected=v[0], api_patch=v[1], prior_use=v[2]) for k, v in C.EXTERNAL_SETS.items()},
                          patch_strings='DB api_patch 16.xx kept verbatim and separate from DB public_patch 26.xx; no remapping',
                          status='previously accessed / unknown-use sets are exploratory TESTs, not untouched',
                          EUW1='0 complete raw pairs: reported as unavailable, never as a completed test',
                          adaptation='raw detail/timeline JSON -> cache files written inside this output root using the '
                                     'original repository cache builder functions (data.cache_io prebuild path: '
                                     'parse_timeline_to_minute_cache, build_anchors_from_events, static meta, same '
                                     'FEATURE_VERSION) and loaded with the same load_match_cache; no winner/target '
                                     'performance used; raw SHA-256 re-verified; key availability audited; structural '
                                     'inability = explicit set blocker while the main run continues')),
        partitions=dict(
            train_fold="int(sha256('full-v-oof-20260915:'+match_id).hexdigest()[:8],16) % 5; all states/engagements of a match share the fold",
            validation_role="int(sha256('full-val-20260915:'+match_id).hexdigest()[:8],16) % 4 -> 0 V_CAL, 1 V_SELECT, 2 Q_CAL, 3 Q_SELECT",
            usage=dict(TRAIN='V base fit (final: all folds; OOF: other four folds); q fit rows',
                       V_CAL='V calibration candidates only', V_SELECT='V calibration choice only',
                       Q_CAL='q calibrators only', Q_SELECT='q candidate choice only',
                       TEST='frozen evaluation only (15.16)', external='frozen evaluation only, per set'),
            diagnostics='q performance on V_CAL/V_SELECT and V performance on Q_CAL/Q_SELECT are diagnostic only, '
                        'computed after the corresponding selection is frozen'),
        v_model=dict(
            target='W = final Blue win from GAME_END winningTeam (gameplay.state_value.final_outcome), stored in separate '
                   'outcome files, never a feature',
            state='objective_history_v2_participant_order via gameplay.state_value_v2.StateBuilder; observation-causal '
                  '(latest frame <= query, predictor events <= query); participant order team then participantId; '
                  'no positions, no future interpolation, no postgame role slots; soul type/ownership fixes kept',
            features="expanded family: all StateV2 columns except snapshot_age_s (audit only); champion IDs one-hot "
                     "categorical (OneHotEncoder handle_unknown='ignore'); numeric median impute + StandardScaler",
            estimator='train.state_value_experiment.logistic(cols, C=0.01): LogisticRegression(solver=liblinear, '
                      'max_iter=1500, random_state=7); C=.01 PREDECLARED from the exploratory pilot, not a validated optimum; '
                      'no hyperparameter search',
            fit_weights='equal total weight per match: weights(groups) = (1/rows_in_match)/mean over the fitted rows',
            convergence='record n_iter_ and every ConvergenceWarning; non-convergence is reported, not hidden',
            query_sampling=dict(
                grid='t = 120000 + k*60000 ms with first_frame <= t <= last_frame and t < terminal (min GAME_END timestamp)',
                buckets='fixed 5-minute buckets floor(t/300000); one query per nonempty bucket',
                choice="minimum sha256('<match_id>:<query_ms>:full-v-query-20260915') hex digest within the bucket",
                weights='equal total weight per match in fitting, calibration, selection and evaluation',
                disclosure='the grid upper bound uses the observed terminal time retrospectively: this is a TRAINING/'
                           'evaluation sampling scheme, not an inference input; no duration/time-to-end predictor',
                test_secondary='TEST and external: full minute trajectories (all grid queries) as secondary evaluation '
                               'with equal match weights'),
            calibration=dict(candidates=['raw (identity)', 'sigmoid_pos: expit(a + b*logit(clip(p,1e-12,1-1e-12))), b >= 1e-6, '
                                         'weighted MLE by scipy.optimize.minimize L-BFGS-B on V_CAL with match weights'],
                             isotonic='NOT a V candidate (flat regions can erase delta): documented design choice',
                             selection='lowest V_SELECT match-weighted log loss, then Brier, then name',
                             freeze='selection_v.json with model hashes written before OOF labels, q fitting or TEST'),
            oof=dict(folds=5, rule='fold-k base V fitted only on TRAIN matches of the other four folds (preprocessing, '
                                   'vocabulary and scaling refit inside); if the final V selected sigmoid_pos, each fold '
                                   'model gets its own sigmoid_pos fitted on V_CAL with the same rule; otherwise raw',
                     use='every TRAIN engagement p_pre/p_post (and q input p_pre_V) comes from its held-out-fold adapter; '
                         'validation/test/external engagements use the final V',
                     audits=['per-row fold and adapter hash', 'fitting-membership: held-out match never in its adapter fit set',
                             'calibration origin', 'held-out TRAIN V metrics vs W', 'fold dispersion on V_SELECT',
                             'OOF-vs-final label agreement on TRAIN (diagnostic only; final-V TRAIN labels never used for q)'],
                     statement='cross-fitting of generated targets, not proof of semantic truth')),
        engagements=dict(
            detector='parent v3.3 detector (original repository gameplay.fights.detect_fights + data.index_split._fight_to_ref_row): '
                     'TF2_KILL_CLUSTER_GAP_MS=13700 (G 13.7 s), CLUSTER_MAX_DIAMETER=4264 (D), TF2_VALIDITY_RADIUS=1600 (R), '
                     'TF2_ENGAGE_PRE_KILL_MS=15000 (B 15 s), TF2_MIN_PER_TEAM=2; checked against core/presets.py and config.py',
            main='parent exposures (not re-detected); external: same frozen detector run on adapted caches, exposures built by '
                 'measure_postkill_objective_delay.analyse_match with the same sentinel semantics',
            times='s = first kill - 15000 ms; q_pre = s - 1 ms; L = last kill',
            endpoint='endpoint_h = min(L + 1000h, next raw CHAMPION_KILL strictly after L - 1, next eligible (stored next_start) '
                     'engagement start - 1 (absent if next_start >= game end), game_end - 1) for h = 60, 90, 120 s',
            validity='endpoint >= L, endpoint > q_pre, endpoint <= last frame, q_pre within observed frames, next_start > L, '
                     'StateV2 builds at both queries; failures excluded with reason lineage',
            objectives='objectives never terminate windows',
            horizons=dict(primary=90, sensitivity=[60, 120], rule='never optimized on TEST'),
            labels='delta = V(post) - V(pre) with the SAME adapter; Y = 1[delta > 0]; exact zero -> 0 (reported)',
            saved=['all boundaries and tied reasons', 'validity/exclusion reasons', 'pre/post snapshot timestamps and ages',
                   'post snapshot after L flag (staleness)', 'raw kills in (L, endpoint]', 'endpoint equality across horizons',
                   'adapter/model hashes and fold', 'raw event counts during (q_pre, L] and after (L, endpoint] (post-hoc strata only)'],
            not_predictors=['L', 'duration', 'endpoint/end reason', 'post features', 'future roster', 'W'],
            semantics='observational change in model-estimated win probability, not a causal engagement effect'),
        objective_audit='census of ELITE_MONSTER_KILL monsterType/subType, DRAGON_SOUL_GIVEN fields and BUILDING_KILL types '
                        'per role/set; StateV2 maps BARON_NASHOR, RIFTHERALD, HORDE (grubs), ATAKHAN, DRAGON '
                        '(elements AIR/EARTH/FIRE/WATER/HEXTECH/CHEMTECH, other->OTHER; ELDER_DRAGON->elder), soul ownership; '
                        'any other monsterType is unsupported (not counted by StateV2) and reported, never claimed supported',
        q_model=dict(
            inputs='StateV2 at q_pre minus snapshot_age_s, plus p_pre_V; champion IDs excluded from numeric q; no positions',
            sets='ridge = all non-champion inputs (expected 352 incl. p_pre_V, verified at run time); economic = restored P4 '
                 'econ_rule (participant_slot economy/combat + team aggregates + time_minutes); p_pre = [p_pre_V]',
            rows='TRAIN rows valid at h train; Q_CAL calibrates; Q_SELECT selects; equal match weights within each subset',
            candidates=dict(constant='TRAIN match-weighted positive rate',
                            p_pre_logistic='StandardScaler + LogisticRegression(C=1, max_iter=2000)',
                            p_pre_spline='SplineTransformer(n_knots=5, degree=3) + StandardScaler + LogisticRegression(C=1, max_iter=2000)',
                            ridge_raw='StandardScaler + LogisticRegression(C=0.01, max_iter=3000)',
                            ridge_sigmoid='LogisticRegression(C=1e6, max_iter=1000) on logit(clip(raw,1e-8)) fitted on Q_CAL',
                            ridge_isotonic="IsotonicRegression(out_of_bounds='clip') on raw fitted on Q_CAL",
                            economic_raw='mean of LGBMClassifier seeds 7/42/123, 250 trees, 15 leaves, lr .04, min_child_samples 100, '
                                         'reg_lambda 1, colsample_bytree .9, n_jobs 4',
                            economic_sigmoid='as ridge_sigmoid', economic_isotonic='as ridge_isotonic'),
            selection='lowest Q_SELECT match-weighted Brier, then log loss, then name; saved with bundle hashes for h90, h60, '
                      'h120 before any TEST label or prediction',
            order='h90 main first; h60 and h120 trained independently afterwards as sensitivity',
            disclosure='pilot isotonic clipping failure (P4 A90: q=1 rows with y=0) disclosed; not fixed using new TEST',
            pilot_comparison='old pilot q numbers are non-comparable (labeler/scale/cohort change), never claimed as improvement'),
        evaluation=dict(
            v='V vs W: AUC/Brier/log loss (sklearn; log loss clips at float64 eps, saved probabilities never altered), '
              'calibration slope/intercept, calibration-in-the-large, ECE, reliability bins; primary on bucket queries, '
              'secondary on full minute trajectories; time bands 2-10/10-20/20-30/30+ and P1 objective-history strata with N; '
              'TEST and each external set separately; V_SELECT/V_CAL, OOF TRAIN and Q roles as diagnostics',
            q='q vs generated Y per horizon: same metrics; start-time bands, pre-history objective strata, post-hoc during/after '
              'event strata (descriptive); TEST and each external set separately; V_CAL/V_SELECT diagnostic',
            bootstrap='1000 match-resampling replicates (seed 20260915), fixed models: V selected vs TRAIN prior constant and vs '
                      'the other calibration candidate; q chosen vs constant, p_pre_logistic, p_pre_spline, selected economic, '
                      'selected ridge (identical pairs skipped)',
            counts='raw matches, loaded, valid V matches/states, engagement matches/rows and exclusion reasons for every role/set',
            sparse='cells with < 30 matches flagged; single-class AUC/calibration NA',
            freeze='all selections and model hashes frozen (frozen_manifest.json) before TEST/external labels are generated or '
                   'scored; hashes re-verified after evaluation; serialized objects must reproduce saved predictions'),
        shap=dict(target='chosen h90 q', explained='512 TEST rows by sha256 rank', background='256 TRAIN rows by sha256 rank',
                  method='restored P4 methods: exact linear interventional SHAP on the declared log-odds scale; tree models '
                         'LightGBM pred_contrib on raw log-odds labelled base score; exact group-level interventional Shapley '
                         'on the final output (2^G coalitions); additivity identity checks; output scale disclosed',
                  reference='Lundberg & Lee (2017)', not_causal=True),
        temporal_comparator=dict(
            priority='after primary V -> OOF labels -> q -> all tests -> SHAP; never a prerequisite; never replaces the primary labeler',
            model='P2 CandidateB (train/temporal_history_winprob_v3.py): 9 one-minute history positions ending at the query, '
                  'SimpleRNN 8 tanh units, input dropout .25, RMSprop lr .001, batch 64, 50 epochs, seeds 17/29/43, CPU torch <= 4 threads',
            data='same full TRAIN bucket query keys and causal history; fit-only preprocessing (numeric scaling, champion vocabulary)',
            calibration='raw and positive-slope sigmoid per seed on V_CAL, chosen by V_SELECT log loss; fixed mean ensemble',
            report='independent V performance and trajectories against primary; reported separately if pending'),
        verification=['contract tests on synthetic and TRAIN-only fixtures before full fit (split membership, OOF self-outcome '
                      'exclusion, future perturbation invariance, causal boundaries, no postgame predictors, transform fit '
                      'origins, serialization, independent metric reconstruction)', 'smoke runs use TRAIN matches only',
                      'protocol before fitting; selections before TEST; lineage/objective availability audits; convergence '
                      'records; model hashes before TEST and after'],
        literature=dict(
            Maymin2021='DOI 10.1515/jqas-2019-0096: snapshot state win probability / value',
            Kim2020='CoG 2020 confidence-calibrated MOBA winner predictor: probability/calibration motivation; uncertainty loss not reproduced',
            Hodge2019_2021='IEEE ToG 13(4):368-379, DOI 10.1109/TG.2019.2948469: live prediction motivation only',
            Silva2018='SBGames 2018 RNN continuous outcome prediction: temporal comparator architecture, adapted',
            LundbergLee2017='SHAP (NeurIPS 2017, arXiv:1705.07874)',
            Jalovaara2024='master thesis, 5-minute bucket sampling; not peer-reviewed',
            our_choices='five folds, validation hash partitions, positive-slope calibration and 90 s are protocol choices of '
                        'this project, not established by these citations'),
        claims_not_made=['labels are ground truth', 'causal engagement effects', 'player skill', 'untouched external tests for '
                         'previously accessed sets', 'all-patch generalization', 'production readiness'],
        outputs='outputs/full_corpus_training_20260915 only (plus new scripts/fc20260915_*.py and tests/test_fc20260915_*.py)')
    protocol['protocol_sha256_of_content'] = C.sha256_json(protocol)
    digest = C.write_json(path, protocol)
    print(json.dumps(dict(written=str(path), sha256=digest, detector_sources_all_equal=all(v['equal'] for v in detector_sources.values()))))


if __name__ == '__main__':
    main()
