"""Write AUDIT_CORRECTION_RECEIPT.json for the incremental_q_training_20260915 audit correction (read-only checks + one output file)."""
import hashlib
import json
import pathlib
import time

A = pathlib.Path(__file__).resolve().parent
O = A.parent
R = O.parents[1]
H = A / 'history' / 'pre_correction_v1'


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


fz = load(O / 'frozen_manifest.json')
res = load(O / 'eval' / 'results.json')
hist = load(H / 'manifest.json')
ver = load(R / 'outputs' / 'claude_dispatch_incremental_q' / 'independent_result_verification.json')
tss = load(R / 'outputs' / 'claude_dispatch_incremental_q' / 'training_source_snapshot' / 'manifest.json')
val_new, val_old, cand = load(O / 'validation.json'), load(H / 'run' / 'validation.json'), load(A / 'validation_candidate_run1.json')
fdiff = load(O / 'integrity' / 'snapshot_final_diff.json')
frozen_now = {p: sha(O / p) == s for p, s in fz['frozen_files_sha256'].items()}
eval_preds = {f'{s}/{c}': sha(O / res['results'][s][c]['predictions_file']) == res['results'][s][c]['predictions_sha256']
              for s in res['results'] for c in res['results'][s]}
changed = ['scripts/iq20260915_report.py', 'scripts/iq20260915_postrun_checks.py', 'scripts/iq20260915_fit.py']
gate = cand['details']['bootstrap_acceptance_all_B40_B45_cells']


def last_line(p):
    return pathlib.Path(p).read_bytes().decode('utf-8', errors='replace').strip().splitlines()[-1]


rec = dict(
    receipt='incremental_q_training_20260915 audit correction', written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
    scope='bounded post-run correction; NO refit, reselection, label change, bootstrap regeneration or change to frozen model/prediction/selection artifacts',
    inputs_read=['outputs/claude_dispatch_incremental_q/independent_code_review.json',
                 'outputs/claude_dispatch_incremental_q/independent_result_verification.json',
                 'outputs/claude_dispatch_incremental_q/training_source_snapshot/manifest.json'],
    findings=[
        dict(id=1, file='scripts/iq20260915_report.py',
             finding='hardcoded improvement magnitude and Y baseline Brier ~.25 inferred from p_pre ~.5',
             removed_text=('절대 개선 폭은 Brier 소수 셋째~넷째 자리 수준이다. B40/B45는 p_pre가 .5 근처인 모집단이라 기저 Brier가 0.25에 가깝고, '
                           '구성·유병률·시간대가 달라 전체와의 차이를 "어려움의 인과효과"로 해석하지 않는다. p_pre .5는 V 모델의 추정 균형이지 실제 승률의 정답이 아니다.'),
             change=('replaced by (a) signed ranges of saved MAIN TEST contrast estimates and primary LightGBM-PT estimates relative to PT Brier, '
                     '(b) explicit p_pre = P(final Blue win | S_pre) vs q = P(deltaV>0 | S_pre) distinction (balanced V does not imply balanced Y), '
                     '(c) measured match-weighted Y prevalence and constant-q Brier per cell; the non-causal caution is kept; '
                     'DEFINITION table gains a q row; report section 9 discloses this correction')),
        dict(id=2, file='scripts/iq20260915_postrun_checks.py',
             finding='bootstrap gates checked point estimates only and skipped computed=False cells',
             change=('bootstrap_cell_acceptance() + check bootstrap_acceptance_all_B40_B45_cells: eligibility from saved evaluation prediction cell masks '
                     '(>=30 distinct matches and 2 Y classes); eligible cells require 1000 replicates, seed 20260915, matches/rows equal to the cell, the five '
                     'bootstrap models, the five planned contrasts in protocol order, finite ordered CIs, finite replicates = 1000 (Brier/log loss) and '
                     '1000 - degenerate (AUC), 0 <= degenerate < 1000; ineligible cells require explicit computed=False with a reason; each set/cohort needs '
                     'the explicit time-cell CI-not-computed statement. The contract-test timing gate now selects the latest PASSING contract_tests run '
                     'before the first full fit (run 2) instead of reading result.json. --output lets a candidate validation be written first.'),
             result=dict(eligible_cells_accepted=gate['counts']['eligible'], explicit_skips=gate['counts']['ineligible_explicit_skip'],
                         skipped_cells='EXT_KR_16.14_pilot T B40 and T B45 (<30 matches)', problems=gate['problems'])),
        dict(id=3, file='scripts/iq20260915_fit.py',
             finding='no explicit unique(match, s_ms) assertion; partial-family rerun could overwrite config bundles',
             change=('assert_unique_keys() right after loading, before any fit (SystemExit on duplicates; count stored as pre_fit_unique_row_keys); '
                     'partial_family_artifacts() refusal before loading/fitting when config bundles, predictions or internal_stop files exist without a '
                     'selection file (artifacts retained, actionable message, status failed through the existing handler). No resume framework.'),
             status='CODE-ONLY POST-RUN GUARD; the modified file was NOT the training source and nothing was rerun',
             training_source_evidence=dict(frozen_manifest_source_sha256=fz['source_sha256']['scripts/iq20260915_fit.py'],
                                           dispatch_training_source_snapshot_sha256=tss['files_equal_freeze']['scripts/iq20260915_fit.py'],
                                           preserved_pre_correction_copy='audit_followup/history/pre_correction_v1/scripts/iq20260915_fit.py',
                                           preserved_copy_sha256=hist['files']['scripts/iq20260915_fit.py']),
             root_key_audit='outputs/claude_dispatch_incremental_q/pre_fit_key_audit.json (all 402772 TRAIN+VAL keys unique before training)')],
    source_files={f: dict(at_freeze=fz['source_sha256'].get(f), before_correction=hist['files'][f], after_correction=sha(R / f)) for f in changed},
    post_freeze_code_changes_disclosed=dict(
        note=('Only report, snapshot, postrun_checks and fit scripts differ from the freeze-time sources. report/snapshot/postrun_checks had already been '
              'edited after the freeze in the original run (report wording, final integrity phase, ordering and legacy checks); fit was edited only in this '
              'correction. None of these edits produced or changed models, selections or predictions.'),
        unchanged_freeze_sources={f: sha(R / f) == s for f, s in fz['source_sha256'].items() if f not in changed + ['scripts/iq20260915_snapshot.py']},
        snapshot_py=dict(at_freeze=fz['source_sha256']['scripts/iq20260915_snapshot.py'], now=sha(R / 'scripts' / 'iq20260915_snapshot.py'))),
    frozen_artifacts_verified_now=dict(all_equal=all(frozen_now.values()), files=len(frozen_now)),
    eval_results_sha256=dict(now=sha(O / 'eval' / 'results.json'), root_verified=ver['results_sha256'],
                             equal=sha(O / 'eval' / 'results.json') == ver['results_sha256']),
    eval_prediction_files_equal_recorded=all(eval_preds.values()),
    tests=dict(file='audit_followup/test_iq20260915_audit_followup.py', sha256=sha(A / 'test_iq20260915_audit_followup.py'),
               runs=[dict(run=1, log='audit_followup/tests_run1.txt', result=last_line(A / 'tests_run1.txt')),
                     dict(run=2, log='audit_followup/tests_run2.txt', result=last_line(A / 'tests_run2.txt'),
                          note='rerun after the final p_pre wording edit of report.py')],
               not_a_prerequisite='audit tests are outside contract_tests/; the timing gate uses contract_tests run 2 (before the first full fit)'),
    validation=dict(
        before=dict(file='audit_followup/history/pre_correction_v1/run/validation.json', sha256=hist['files']['run/validation.json'],
                    n_checks=val_old['n_checks'], failed=val_old['failed']),
        candidate=dict(file='audit_followup/validation_candidate_run1.json', sha256=sha(A / 'validation_candidate_run1.json'),
                       n_checks=cand['n_checks'], failed=cand['failed']),
        final=dict(file='validation.json', sha256=sha(O / 'validation.json'), n_checks=val_new['n_checks'], failed=val_new['failed'],
                   added_checks=sorted(set(val_new['checks']) - set(val_old['checks'])),
                   removed_checks=sorted(set(val_old['checks']) - set(val_new['checks'])))),
    report=dict(before=dict(REPORT_md=hist['files']['run/REPORT.md'], DEFINITION_AND_EVIDENCE_md=hist['files']['run/DEFINITION_AND_EVIDENCE.md']),
                after=dict(REPORT_md=sha(O / 'REPORT.md'), DEFINITION_AND_EVIDENCE_md=sha(O / 'DEFINITION_AND_EVIDENCE.md'))),
    parent_integrity_after_correction=dict(file='integrity/snapshot_final_diff.json', all_equal=fdiff['all_equal'],
                                           previous_final_diff_preserved='audit_followup/history/pre_correction_v1/run/integrity/'),
    sealed_access_note=('the candidate and final post-run check runs re-opened MAIN TEST after the freeze for the existing reload-identity check '
                        '(logged in access_log.jsonl); no fitting, selection or bootstrap computation'),
    history='audit_followup/history/pre_correction_v1/manifest.json')
assert rec['frozen_artifacts_verified_now']['all_equal'] and rec['eval_results_sha256']['equal'] and rec['eval_prediction_files_equal_recorded']
assert not val_new['failed'] and all(rec['post_freeze_code_changes_disclosed']['unchanged_freeze_sources'].values()) and fdiff['all_equal']
data = json.dumps(rec, indent=1, ensure_ascii=False).encode('utf-8')
(A / 'AUDIT_CORRECTION_RECEIPT.json').write_bytes(data)
print('receipt sha256', hashlib.sha256(data).hexdigest())
print(json.dumps(dict(validation=rec['validation']['final'], frozen=rec['frozen_artifacts_verified_now'], results=rec['eval_results_sha256']), ensure_ascii=False))
