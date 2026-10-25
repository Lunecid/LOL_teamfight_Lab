r"""[EPT-T002 전용 학습 오퍼레이터: training_operator.py]

한국어 개요 및 하네스 위치:
이 모듈은 EPT-T002 승인 작업서에 따른 단일 학습 실행(Check, Start, Status)을 관장하는 안전 오퍼레이터입니다.
- 지원 액션:
  1) Check: 읽기 전용으로 Anaconda Python, 입력 스토어(S1, S2, S3), 설정 파일 SHA-256 해시, E6 가드를 점검.
  2) Start: 사전에 엄격히 검증된 단일 프로필(S 모델, CUDA, 25% 서브셋, patience 2, 최대 30 에폭) 1회 실행 시작.
  3) Status: 실행 디렉토리의 PROCESS.json 및 RUN.json을 읽어 생존 여부와 단계별 상태를 안전하게 조회.
- 제약 및 불변 원칙:
  - 재시도(Resume) 기능은 제공하지 않으며, 단 1회의 완료/실패 후 안전하게 종료합니다.
  - D:\LOL_Data\runs 외부 경로로의 실행 생성이나 승인되지 않은 하이퍼파라미터 변경을 차단합니다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import re
import sys

from c_ept.workflows import s4a_operator as common
from c_ept.workflows.s4a_runtime import RuntimeRefused, prepare_native_runtime

ROOT = common.ROOT
PYTHON = Path(r'C:\Users\todtj\anaconda3\python.exe')
RUNS_ROOT = Path(r'D:\LOL_Data\runs')
RUN_NAME = re.compile(r'^\d{8}_c_ept_ept_pilot_\d+$')
STAGES = ['scales', 'stats', 'train_S_cuda']
RUN_SCHEMA = 'ept_training_run_v1'
PROCESS_SCHEMA = 'ept_training_process_v1'
FIXED = {'schema': 'ept-training-operator-v1', 'profile': 's25_early_stop',
         'size': 'S', 'device': 'cuda', 'max_epochs': 30, 'share': .25,
         'patience': 2, 'batch_size': 32, 'seed': 20261022, 'lr': .001,
         'weight_decay': .01, 'warmup_steps': 10}
PATH_KEYS = ('python', 's1', 's2', 's3', 'runs_root', 'catalog_db')
OperatorError = common.OperatorError
read_json = common.read_json
config_digest = common.config_digest


def load_config(path, *, require_inputs=True):
    """지정된 경로에서 학습 설정 JSON을 로드하고 스키마 및 고정 하이퍼파라미터를 검증한다."""
    c = read_json(path)
    if set(c) != set(FIXED) | set(PATH_KEYS):
        raise OperatorError('training config fields/schema mismatch')
    for key, expected in FIXED.items():
        value = c[key]
        if type(expected) is float:
            valid = type(value) in (int, float) and math.isfinite(value) and value == expected
        else:
            valid = type(value) is type(expected) and value == expected
        if not valid:
            raise OperatorError(f'authorized profile requires {key}={expected}')
    paths = {key: common.canonical_absolute(c[key], key) for key in PATH_KEYS}
    if paths['python'] != PYTHON.resolve():
        raise OperatorError('fixed Anaconda Python required')
    if paths['runs_root'] != RUNS_ROOT.resolve():
        raise OperatorError('runs_root must be D:\\LOL_Data\\runs')
    if not paths['catalog_db'].is_relative_to(paths['runs_root'].parent):
        raise OperatorError('catalog_db must remain under D:\\LOL_Data')
    if require_inputs:
        if not paths['python'].is_file() or paths['python'] != Path(sys.executable).resolve():
            raise OperatorError('configured fixed Python differs from current interpreter')
        for key in ('s1', 's2', 's3'):
            if not paths[key].is_dir():
                raise OperatorError(f'{key} input directory unavailable')
    return c


def config_path(value):
    path = common.canonical_absolute(str(value), 'config')
    if path.is_relative_to(ROOT):
        raise OperatorError('operational config must be external to source checkout')
    return path


def control_path(value, c):
    if value is None:
        raise OperatorError('Start/Status requires an explicit --control directory')
    path = common.canonical_absolute(str(value), 'control')
    root = Path(c['runs_root'])
    if path == root or not path.is_relative_to(root) or path.is_relative_to(ROOT):
        raise OperatorError('control must be external to source and under approved runs_root')
    # Do not let an old run directory become a new control directory.
    if any(RUN_NAME.fullmatch(part) or re.fullmatch(r'\d{8}_c_ept_.+_\d+', part)
           for part in path.relative_to(root).parts):
        raise OperatorError('control cannot be within a scientific run')
    return path


def preflight(c):
    """연구 페이로드를 열지 않고 메타데이터, Git 커밋, GPU 및 E6 스트라타 가드만 사전 점검한다."""
    common.runtime_check()
    commit = common.code_check()
    common.gpu_metadata()
    for key in ('s1', 's2'):
        path = Path(c[key])
        if (path / key / 'STORE.json').is_file():
            path /= key
        doc = read_json(path / 'STORE.json')
        if doc.get('complete') is not True or type(doc.get('files')) is not dict:
            raise OperatorError(f'{key} store metadata incomplete')
    if read_json(Path(c['s3']) / 'RUN.json').get('stage') != 's3':
        raise OperatorError('s3 stage metadata mismatch')
    read_json(Path(c['s3']) / 'STAGES.json')
    common.e6_check()
    return commit


def validate_run_metadata(record, path):
    # Reuse the strict canonical code/config/runtime/identity validator unchanged.
    if record.get('schema') != RUN_SCHEMA or record.get('stage') != 'ept_training_single':
        raise OperatorError('training RUN schema/stage mismatch')
    projected = dict(record, schema='s4a_run_v1', stage='s4a')
    identity, digest = common.validate_run_metadata(projected, path)
    if (record.get('stages') != STAGES or type(record.get('results')) is not dict
            or not set(record['results']).issubset(STAGES)):
        raise OperatorError('training RUN stage selection mismatch')
    args = identity['args']
    if set(args) != set(FIXED) | set(PATH_KEYS):
        raise OperatorError('training RUN effective args mismatch')
    for key, expected in FIXED.items():
        value = args.get(key)
        numeric = type(expected) is float
        if ((type(value) not in (float, int) or not math.isfinite(value)) if numeric
                else type(value) is not type(expected)) or value != expected:
            raise OperatorError('training RUN profile mismatch: ' + key)
    for key in PATH_KEYS:
        common.canonical_absolute(args.get(key), 'RUN args.' + key)
    if config_digest(args) != identity['operator_config_sha256']:
        raise OperatorError('training RUN config digest mismatch')
    return identity, digest


def validate_process_metadata(record, run, identity, digest):
    if record.get('schema') != PROCESS_SCHEMA:
        raise OperatorError('training PROCESS schema mismatch')
    common.validate_process_metadata(dict(record, schema='s4a_process_v1'), run, identity, digest)


def bound_run(control, c):
    """컨트롤 디렉터리의 run.json 참조로부터 실행 디렉터리, 메타데이터, identity를 역참조 검증한다."""
    ref = read_json(control / 'run.json')
    if set(ref) != {'run', 'code_commit', 'operator_config_sha256', 'run_identity_sha256'}:
        raise OperatorError('control run ref fields mismatch')
    run = common.canonical_absolute(ref['run'], 'control run')
    if run.parent != Path(c['runs_root']) or not RUN_NAME.fullmatch(run.name) or not run.is_dir():
        raise OperatorError('control run ref outside approved pilot family')
    record = read_json(run / 'RUN.json')
    identity, digest = validate_run_metadata(record, run)
    if (ref['run_identity_sha256'] != digest or ref['code_commit'] != record['code_commit']
            or ref['operator_config_sha256'] != config_digest(c)
            or record['operator_config_sha256'] != ref['operator_config_sha256']
            or identity['args'] != c):
        raise OperatorError('control ref/code/config/RUN binding mismatch')
    return run, record, identity, digest


def status(control, c):
    """실행 디렉터리의 STAGES.json 및 PROCESS.json을 읽어 현재 학습 진행률과 프로세스 상태를 반환한다."""
    run, manifest, identity, digest = bound_run(control, c)
    result = {'run': str(run), 'run_identity_sha256': digest, 'state': 'unknown'}
    if (run / 'STAGES.json').exists():
        ledger = read_json(run / 'STAGES.json')
        stages = ledger.get('stages')
        if (ledger.get('schema') != 's3_stages_v1' or type(stages) is not dict
                or not set(stages).issubset(STAGES) or type(ledger.get('calls')) is not list):
            raise OperatorError('training stage ledger mismatch')
        completed = []
        incomplete = False
        for name in STAGES:
            stage = stages.get(name)
            if stage is None:
                incomplete = True
                if name in manifest['results']:
                    raise OperatorError('RUN result missing its stage ledger')
                continue
            if type(stage) is not dict or type(stage.get('complete')) is not bool:
                raise OperatorError('stage completion type mismatch')
            expected = config_digest({'identity': identity, 'stage': name})
            if (stage.get('inputs_sha256') != expected
                    or stage.get('inputs') != {'run_identity_sha256': digest}):
                raise OperatorError('stage input identity mismatch')
            if stage['complete']:
                if incomplete:
                    raise OperatorError('completed stage after incomplete stage')
                path = run / (name + '.json')
                record = read_json(path)
                outputs = stage.get('outputs')
                if (type(outputs) is not dict or not outputs
                        or any(not isinstance(k, str) or Path(k).is_absolute() or '..' in Path(k).parts or ':' in k
                               for k in outputs)
                        or record.get('schema') != 's4a_stage_v1' or record.get('stage') != name
                        or record.get('inputs_sha256') != expected
                        or manifest['results'].get(name) != record):
                    raise OperatorError('completed stage result/ledger binding mismatch')
                import hashlib
                if outputs.get(name + '.json') != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise OperatorError('stage root output hash mismatch')
                completed.append(name)
            else:
                incomplete = True
        result['complete_stages'] = completed
    elif manifest['results']:
        raise OperatorError('RUN results without STAGES ledger')
    process = read_json(run / 'PROCESS.json')
    validate_process_metadata(process, run, identity, digest)
    evidence = common.observe_process(process['pid'])
    result['process_evidence'] = evidence['kind']
    result['process_state'] = process['state']
    result['recorded_exit_code'] = process['exit_code']
    same = evidence['kind'] == 'observed' and evidence.get('create_time') == process['create_time']
    if process['state'] == 'running':
        result['state'] = 'running' if same else 'interrupted_unconfirmed' if evidence['kind'] == 'missing' else 'unknown'
    else:
        result['state'] = 'recorded_completion' if process['state'] == 'finished' else 'recorded_failure'
        result['owner_observed_alive'] = same
        if process['state'] == 'finished' and result.get('complete_stages') != STAGES:
            raise OperatorError('finished PROCESS without three complete stages')
    return result


def main(argv=None):
    """EPT-T002 오퍼레이터 CLI 진입점 (Check, Start, Status)."""
    parser = argparse.ArgumentParser(description='One EPT pilot; default read-only Check')
    parser.add_argument('--action', choices=('Check', 'Start', 'Status'), default='Check')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args(argv)
    try:
        policy = prepare_native_runtime() if args.action != 'Status' else None
        path = config_path(args.config)
        c = load_config(path, require_inputs=args.action != 'Status')
        control = control_path(args.control, c) if args.control is not None or args.action != 'Check' else None
        if args.action == 'Status':
            print(json.dumps(status(control, c), ensure_ascii=False, indent=2))
            return 0
        if args.action == 'Check':
            commit = preflight(c)
            print(json.dumps({'state': 'READY_WITHIN_SCOPE', 'code_commit': commit,
                              'operator_config_sha256': config_digest(c), 'runtime_policy': policy,
                              'scope': 'metadata preflight; full input verification at Start'}, indent=2))
            return 0
        runner = importlib.import_module('c_ept.workflows.run_training')
        return runner.main(['--config', str(path), '--control', str(control),
                            '--operator-config-sha256', config_digest(c)])
    except (OperatorError, RuntimeRefused, OSError, ValueError) as exc:
        print('EPT training refused: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
