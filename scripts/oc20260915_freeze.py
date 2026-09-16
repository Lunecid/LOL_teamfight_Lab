"""Objective-channel ablation stage F: freeze all B_noobj adapters, calibrators, memberships and the secondary choice.

Requires passing synthetic pytest, smoke and full model checks, and a complete manifest. Verifies every adapter hash;
confirms pre-freeze outcome access was limited to TRAIN folds / V_CAL / V_SELECT and no sealed label was opened.
"""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import lv20260915_common as LV  # noqa: E402
import oc20260915_common as K  # noqa: E402

import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()] if Path(p).exists() else []


def main():
    st = K.Status('freeze')
    K.log_command()
    fz_path = K.frozen_manifest_path()
    if fz_path.exists():
        raise SystemExit('frozen_manifest.json exists; refusing to overwrite')
    problems = []
    logs = sorted((K.OUT / 'contracts').glob('pytest_run*.txt'))
    last = logs[-1].read_text(encoding='utf-8', errors='replace') if logs else ''
    if not re.search(r'\b\d+ passed\b', last) or re.search(r'\b(failed|error)\b', last):
        problems.append('synthetic pytest not passing')
    for kind in ('smoke', 'full'):
        p = K.OUT / 'contracts' / f'v_model_checks_{kind}.json'
        if not p.exists() or not C.read_json(p)['all_pass']:
            problems.append(f'v_model_checks_{kind} missing or failing')
    man = C.read_json(K.OUT / 'v_models_manifest_B_noobj.json')
    files = {}
    for fam, fd in man['families'].items():
        files[fd['final_path']] = fd['final_sha256']
        for k, sha in fd['oof_sha256'].items():
            files[fd['oof_paths'][k]] = sha
    bad = [p for p, sha in files.items() if C.sha256_file(K.OUT / p) != sha]
    if bad:
        problems.append(f'adapter hash mismatch {bad}')
    if not man['converged_all']:
        problems.append('non-converged fit')
    acc = jsonl(K.OUT / 'outcome_access_log.jsonl')
    allowed = LV.ParentReadOnly.PRE_FREEZE_OUTCOME_ROLES
    viol = [a for a in acc if a['set'] != 'MAIN' or not a['sub_roles'] or not set(a['sub_roles']) <= allowed]
    if viol:
        problems.append(f'pre-freeze outcome access outside TRAIN/V_CAL/V_SELECT: {viol[:3]}')
    lab = jsonl(K.OUT / 'label_access_log.jsonl')
    if [a for a in lab if a['set'] in K.SEALED_LABEL_SETS]:
        problems.append('sealed labels opened before freeze')
    if (K.OUT / 'eval').exists() or (K.OUT / 'labels').exists():
        problems.append('evaluation or label outputs exist before freeze')
    if problems:
        st.update('failed', 'freeze', error='; '.join(problems)[:1500], next_step='resolve')
        raise SystemExit('freeze refused: ' + '; '.join(problems))
    sel_p = K.OUT / 'selection' / 'selection_secondary_calibration.json'
    vman = C.read_json(K.FC / 'v_models_manifest.json')
    fz = dict(role=K.ROLE_TAG, version=K.VERSION, frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
              protocol_sha256=C.sha256_file(K.OUT / 'protocol.json'), feature_evidence_sha256=C.sha256_file(K.OUT / 'feature_evidence.json'),
              manifest_sha256=C.sha256_file(K.OUT / 'v_models_manifest_B_noobj.json'), selection_sha256=C.sha256_file(sel_p),
              primary=man['primary'], secondary=man['secondary'], families=man['families'], adapter_file_sha256=files,
              contracts=dict(pytest_log=logs[-1].name, pytest_log_sha256=C.sha256_file(logs[-1]),
                             v_model_checks_smoke_sha256=C.sha256_file(K.OUT / 'contracts' / 'v_model_checks_smoke.json'),
                             v_model_checks_full_sha256=C.sha256_file(K.OUT / 'contracts' / 'v_model_checks_full.json')),
              primary_A=dict(v_final_sha256=vman['final']['sha256'], v_final_path=vman['final']['path'], v_oof_sha256=vman['oof_sha256'],
                             v_oof_paths=vman['oof_paths']),
              pre_freeze_outcome_access=[dict(time=a['time'], sub_roles=a['sub_roles'], purpose=a['purpose'], rows=a['rows']) for a in acc],
              pre_freeze_label_access=[dict(time=a['time'], set=a['set'], purpose=a['purpose']) for a in lab],
              statement='B_noobj final + 5 OOF (raw primary, both calibration families) and the secondary choice are frozen before any '
                        'TEST/external/Q_CAL/Q_SELECT outcome or TEST/external label access by this study.')
    sha = C.write_json(fz_path, fz)
    st.update('complete', 'freeze', frozen_manifest_sha256=sha, secondary_chosen=man['secondary']['chosen'], next_step='W evaluation, labels')
    return 0


if __name__ == '__main__':
    sys.exit(main())
