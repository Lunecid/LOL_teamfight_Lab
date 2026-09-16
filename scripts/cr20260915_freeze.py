"""Freeze every new selection (A specialists, role models, C arms, ablations) before any TEST / external label is opened."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402


def main():
    st = K.Status('freeze')
    fz_path = K.OUT / 'frozen_manifest.json'
    if fz_path.exists():
        raise SystemExit('frozen manifest already exists')
    if (K.OUT / 'eval').exists():
        raise SystemExit('eval/ exists before freeze')
    log = K.OUT / 'label_access_log.jsonl'
    if log.exists() and any('MAIN_TEST' in l or 'EXT_' in l for l in log.read_text(encoding='utf-8').splitlines() if json.loads(l).get('frozen_manifest_exists') is not False):
        raise SystemExit('label access after a freeze recorded before this freeze')
    files, status = {}, {}
    sel_dir = K.OUT / 'selection'
    spec = {}
    for coh in K.COHORTS:
        for h in K.HS:
            p = sel_dir / f'q_specialist_{coh}_h{h}.json'
            if not p.exists():
                raise SystemExit(f'missing specialist selection {p.name} (A must be complete)')
            s = C.read_json(p)
            spec[f'{coh}_h{h}'] = dict(chosen=s['chosen'], selection_sha256=C.sha256_file(p), written_at=s['written_at'])
            files[str(p.relative_to(K.OUT))] = C.sha256_file(p)
            for cand, sha in s['bundle_sha256'].items():
                files[f'models/q_specialist/{coh}/h{h}/{cand}.joblib'] = sha
    status['A'] = 'selections complete'
    roles = None
    rm = K.OUT / 'role_models_manifest.json'
    if rm.exists():
        r = C.read_json(rm)
        files['role_models_manifest.json'] = C.sha256_file(rm)
        for k, v in r['models'].items():
            files[v['path'].replace('\\', '/')] = v['sha256']
        roles = dict(models={k: dict(sha256=v['sha256'], converged=v['fit_record']['converged'], n_iter=v['fit_record']['n_iter']) for k, v in r['models'].items()},
                     outputs={k: v['sha256'] for k, v in r['outputs'].items()})
        for k, v in r['outputs'].items():
            files[v['file']] = v['sha256']
        status['B'] = 'role models fitted (weak supervision)'
    else:
        status['B'] = 'BLOCKED: no role models'
    arms = {}
    for coh in K.COHORTS:
        p = sel_dir / f'arms_{coh}_h90.json'
        if p.exists():
            s = C.read_json(p)
            arms[coh] = dict(arm_winners=s['arm_winners'], overall=s['overall_choice'], A_specialist=s['A_specialist_choice'],
                             selection_sha256=C.sha256_file(p), written_at=s['written_at'])
            files[str(p.relative_to(K.OUT))] = C.sha256_file(p)
            for cand, sha in s['bundle_sha256'].items():
                files[f'models/arms/{coh}/h90/{cand}.joblib'] = sha
    status['C'] = 'arms selected for ' + ','.join(sorted(arms)) if arms else 'BLOCKED/not run'
    ab = sel_dir / 'ablations_T_h90.json'
    abl = None
    if ab.exists():
        s = C.read_json(ab)
        abl = dict(preselected=s['preselected_full_role_ridge'], primary_calibration=s['primary_calibration'],
                   blocks={g: dict(primary=v['primary_candidate'], own=v['own_q_select_choice']) for g, v in s['blocks'].items()},
                   selection_sha256=C.sha256_file(ab))
        files[str(ab.relative_to(K.OUT))] = C.sha256_file(ab)
        for g, v in s['blocks'].items():
            for cand, sha in v['bundle_sha256'].items():
                files[f'models/ablations/T/h90/{cand}.joblib'] = sha
    mism = {p: dict(expected=sha, actual=C.sha256_file(K.OUT / p)) for p, sha in files.items() if C.sha256_file(K.OUT / p) != sha}
    if mism:
        raise SystemExit(f'artifact hashes differ from their records: {list(mism)[:5]}')
    prior = C.read_json(K.FC / 'frozen_manifest.json')
    fz = dict(version=K.VERSION, role=K.ROLE_TAG, frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
              protocol_sha256=C.sha256_file(K.OUT / 'protocol.json'), cohort_manifest_sha256=C.sha256_file(K.OUT / 'cohorts' / 'cohort_manifest.json'),
              draft_manifest_sha256=C.sha256_file(K.OUT / 'draft' / 'draft_manifest.json'),
              role_supervision_provenance_sha256=C.sha256_file(K.OUT / 'role_supervision_provenance.json'),
              contract_tests_result_sha256=C.sha256_file(K.OUT / 'contract_tests' / 'result.json'),
              stage_status=status, A_specialists=spec, role_models=roles, arms=arms, ablations=abl,
              pooled_reference=dict(prior_frozen_manifest_sha256=C.sha256_file(K.FC / 'frozen_manifest.json'),
                                    q_selections=prior['q_selections'],
                                    chosen_bundle_sha256={h: prior['q_bundle_sha256'][h][d['chosen']] for h, d in prior['q_selections'].items()}),
              frozen_files_sha256=files,
              statement='All new cohort-specialist, role-model, representation-arm and ablation choices are frozen here, before any '
                        'TEST 15.16 or external label is opened by this task. No refit, recalibration or reselection afterwards. '
                        'TEST and external sets were already evaluated for the pooled q in the completed full run (prior exposure).')
    sha = C.write_json(fz_path, fz)
    st.update('complete', 'freeze', frozen_manifest_sha256=sha, files=len(files), stage_status=status, next_step='evaluation')
    print('frozen', sha, len(files), status)


if __name__ == '__main__':
    main()
