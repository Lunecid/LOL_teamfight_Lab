"""Write protocol.json before any model fit: spec hash, input hashes, predeclared rules, pairs and interpretation choices."""
from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402


def main():
    if (K.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json exists')
    for d in ('selection', 'role_models', 'models'):
        if (K.OUT / d).exists():
            raise SystemExit(f'{d}/ exists: protocol must predate fitting')
    ct = C.read_json(K.OUT / 'contract_tests' / 'result.json')
    fz_prior = C.read_json(K.FC / 'frozen_manifest.json')
    proto = dict(
        version=K.VERSION, role=K.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        design='docs/CLAUDE_COHORT_ROLE_TRAIN_20260915.md (Codex); implementation and execution Claude Opus 5',
        spec_sha256=C.sha256_file(K.SPEC),
        inputs=dict(prior_run='outputs/full_corpus_training_20260915 (read-only)', prior_frozen_manifest_sha256=C.sha256_file(K.FC / 'frozen_manifest.json'),
                    prior_q_selections=fz_prior['q_selections'], prior_v_final_sha256=fz_prior['v_final_sha256'],
                    cohort_manifest_sha256=C.sha256_file(K.OUT / 'cohorts' / 'cohort_manifest.json'),
                    draft_manifest_sha256=C.sha256_file(K.OUT / 'draft' / 'draft_manifest.json'),
                    role_supervision_provenance_sha256=C.sha256_file(K.OUT / 'role_supervision_provenance.json'),
                    integrity_snapshot_before_sha256=C.sha256_file(K.OUT / 'integrity' / 'snapshot_before.json'),
                    contract_tests=dict(run=ct['run'], passed=ct['passed'], passed_count=ct['passed_count'])),
        prior_test_exposure=('15.16 TEST and all external sets were evaluated for the pooled q in the completed full run (and earlier work); '
                             'this is an exploratory follow-up, not a fresh confirmatory evaluation. Pooled q TEST metrics are already known.'),
        labels='frozen full-run labels unchanged: TRAIN held-out-fold V, VALIDATION/TEST/external final V; h90 primary, h60/h120 sensitivity; '
               'estimated probability-improvement labels, not ground truth',
        A=dict(
            scale='v3.3 participation cluster_blue/cluster_red from corpus_shards_v33 (external: frozen detector re-run); teamfight_min 4, pick_max 1',
            cohorts='E all rows; T min>=4; N known & min<4; unknown excluded from T/N (none observed: all -1 proven zero)',
            fits='fc20260915_fit_q.fit_candidates unchanged on cohort TRAIN rows; calibrators cohort Q_CAL; choice cohort Q_SELECT (Brier, log loss, name); h90, h60, h120',
            reference='pooled frozen full-run chosen q per horizon (h90 ridge_isotonic, h60 ridge_sigmoid, h120 ridge_isotonic)',
            evaluation='identical T rows: pooled chosen vs T specialist chosen; identical N rows: pooled vs N specialist; inside N also pick (min<=1) and '
                       'skirmish (2..3) rows; equal match weights per cell; paired match bootstrap 1000 (seed 20260915) AUC/Brier/log loss '
                       'differences; calibration slope/intercept/ECE; exact 0/1 and opposite-label counts; every external set separately; '
                       'oracle-cohort routing (post-cutoff membership, NOT deployable) vs pooled on all known rows reported separately'),
        B=dict(
            supervision='WEAK: cached role_slots of 15.14 TRAIN (raw main detail unavailable); detectable ID-fill teams excluded; role_supervision_provenance.json',
            classifier='DraftEncoder(champion one-hot handle_unknown=ignore; unordered spell multi-hot) -> LogisticRegression(C=1, lbfgs, multinomial, max_iter=2000; raise cap only if needed)',
            weights='one row per eligible participant; equal total weight per match', oof='5 models on existing TRAIN folds; TRAIN q rows use their fold model; others final',
            posterior='120 one-to-one assignments, product of clipped probabilities, logsumexp, marginals; entropy, max marginal, unknown flags',
            post_freeze_diagnostics='external raw teamPosition role reliability; proxy mechanism audit (get_role_slots vs teamPosition) on external raw'),
        C=dict(
            arms=dict(participant='ridge set 352 numeric; ridge C=.01; NEW full-feature LightGBM (250 trees, 15 leaves, lr .04, min child 100, lambda 1, colsample .9, seeds 7/42/123)',
                      draft='participant set + per-slot champion one-hot + per-slot unordered spell multi-hot (fit on cohort TRAIN); ridge: numeric standardized, indicators unscaled',
                      role='global (191 non-participant incl. time) + p_pre_V + role block (2x5x16) + diffs (5x16) + diff x time_minutes for gold/xp/level (5x3) + team uncertainty (2x7)'),
            calibration='raw / sigmoid (Platt on logit, Q_CAL) / isotonic (Q_CAL)', within_arm='cohort Q_SELECT Brier, log loss, name',
            overall='same rule over the three arm winners and the unchanged A specialist choice',
            ablations='T only: drop each role block (blue/red features, diffs, phase interactions); refit ridge + calibrators; primary variant = calibration preselected for full role ridge on T; own Q_SELECT choice recorded',
            pairs_per_cohort_and_set=['role_winner - participant_winner', 'draft_winner - participant_winner', 'role_winner - draft_winner',
                                      'overall_winner - A_specialist_choice', 'A_specialist_choice - pooled_chosen'],
            ablation_pairs='drop_role_r primary - preselected full role ridge (T, each set with >= 20 matches)'),
        shap=dict(model='each cohort role-arm winner (explains the role model even if the overall winner differs)', groups=list(K.ROLE_GROUPS), coalitions=128,
                  explain="256 TEST 15.16 h90-valid cohort rows by sha256('cr20260915_shap_explain:<cohort>:<match>:<s>')",
                  background="128 cohort TRAIN h90-valid rows by sha256('cr20260915_shap_background:<cohort>:<match>:<s>') (TRAIN inputs with OOF role posteriors)",
                  output='final selected calibrated probability', checks='sum phi + base = prediction; reload identity',
                  summaries='global mean |phi| with 1000 simple bootstrap over explained rows (seed 20260915), signed local examples, time bands with N'),
        freeze='frozen_manifest.json after all specialist, role-model, arm and ablation selections; TEST/external labels opened only afterwards (label_access_log.jsonl)',
        interpretation_choices=[
            'Weak role annotation from cached role_slots (spec fallback clause) because raw main detail is not on disk.',
            'Participant slot representation for arms = frozen ridge set (includes p_pre_V and all non-champion numeric pre inputs).',
            'Draft-control ridge: indicators left unscaled (as the V model one-hot block); numeric columns standardized.',
            'Role uncertainty summaries per team: assignment entropy, mean/min max marginal, champion missing/unseen counts, spell missing/unseen counts.',
            'Unseen flags are relative to the generating role model vocabulary (OOF model for TRAIN rows, final model otherwise).',
            'Ablation primary comparison fixes the calibration family of the preselected full role ridge; each ablation still fits its own Q_CAL calibrator.',
            'Picks/skirmishes inside N are evaluated with the N specialist and the pooled q (no separate pick/skirmish models).',
        ])
    sha = C.write_json(K.OUT / 'protocol.json', proto)
    print('protocol sha256', sha)


if __name__ == '__main__':
    main()
