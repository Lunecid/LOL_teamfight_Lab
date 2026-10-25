"""[EPT-T002 단일 파일럿 학습 실행기: run_training.py]

한국어 개요 및 하네스 위치:
이 모듈은 EPT-T002 승인 작업서에 따라 허가된 단일 프로필(S 모델, CUDA, 25% 데이터, patience 2, 최대 30 에폭)의
학습 파이프라인을 1회 실행하는 전용 실행기(runner)입니다.
- 실행 단계:
  1) scales: S3 가격 및 보상 기반 가치 스케일링 계수 산출 (scales.json)
  2) stats: 전체 FIT 데이터 기반 피처 표준화 통계량 산출 (fit_stats.json)
  3) train_S_cuda: 고정된 단일 학습 파이프라인(c_ept.ept.train.train) 실행
- 제약 및 불변 원칙:
  - 오직 승인된 새 학습 오퍼레이터(training_operator) 설정 규약만을 지원합니다.
  - 재시도(Resume), 비용 외삽(projection), DEV 추론, 가치 합성(composition), 홀드아웃(holdout) 평가는 엄격히 금지됩니다.
  - 실행 전 E6 스트라타 가드(배경 모델 서버 탐지) 및 OS 파일 락(operator_start.once)을 통해 1회 단일 실행을 보장합니다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from c_ept.workflows import run_s4a as primitive
from c_ept.workflows import training_operator as operator
from c_ept.workflows.s4a_runtime import prepare_native_runtime, validate_e6_probe, RuntimeRefused

OPERATOR_API_VERSION = 1


def _process_record(run, manifest, state, outcome=None, exit_code=None):
    """프로세스 상태 기록 딕셔너리를 생성하여 반환한다."""
    record = primitive._process_record(run, manifest, state, outcome, exit_code)
    record['schema'] = operator.PROCESS_SCHEMA
    return record


def _reserve(control, c):
    """상위 호출자의 OS 바이트 락 하에서 단 1회의 실행 시도 토큰(operator_start.once)을 영구 등록한다."""
    for name in ('run.json', 'run.json.tmp', 'operator_start.once'):
        if (control / name).exists():
            raise RuntimeRefused('control has a prior operator attempt/ref; retry and Resume forbidden')
    with (control / 'operator_start.once').open('x', encoding='utf-8') as handle:
        json.dump({'schema': 'ept_training_operator_start_v1', 'pid': os.getpid(),
                   'operator_config_sha256': operator.config_digest(c),
                   'created': dt.datetime.now(dt.timezone.utc).isoformat()}, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def _verify_final_head(folder, c):
    """학습 완료 후 페이로드 없는 리더로 최종 커밋된 헤드 메타데이터 및 프로필 무결성을 검증한다."""
    from dataclasses import asdict
    from c_ept.ept import checkpoints
    latest = checkpoints.latest_metadata(folder)
    if latest is None:
        raise RuntimeRefused('single fit returned without a committed final head')
    _, head, history = latest
    identity = head.get('identity', {})
    expected = {'patience': c['patience'], 'epochs': c['max_epochs'], 'seed': c['seed'],
                'lr': c['lr'], 'weight_decay': c['weight_decay'], 'warmup_steps': c['warmup_steps'],
                'encoder': asdict(primitive._model_config(c['size'])), 'device': c['device'],
                'dtype': 'torch.float32', 'threads': 8, 'deterministic_algorithms': True}
    if (history is None or head.get('done') is not True
            or any(identity.get(key) != value or type(identity.get(key)) is not type(value)
                   for key, value in expected.items())):
        raise RuntimeRefused('final v6 head/profile identity mismatch')
    for name, role, training in (('fit', 'FIT', True), ('stop', 'STOP', False)):
        batches = identity.get(name)
        if (type(batches) is not dict or batches.get('role') != role
                or batches.get('training') is not training
                or type(batches.get('batch_size')) is not int or batches['batch_size'] != c['batch_size']
                or type(batches.get('seed')) is not int or batches['seed'] != c['seed']):
            raise RuntimeRefused('final head FIT/STOP batch identity mismatch')
    return history


def execute(c, control, policy):
    """실행 전 사전 점검을 수행하고 1회 단일 학습 시도를 예약 및 실행한다."""
    # All scientific imports occur after main's shared native bootstrap.
    operator.preflight(c)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '0'):
        raise RuntimeRefused('approved CUDA_VISIBLE_DEVICES=0 required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
        raise RuntimeRefused('incompatible CUBLAS_WORKSPACE_CONFIG')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    args = SimpleNamespace(**c, epochs=c['max_epochs'])
    control = operator.control_path(control, c)
    # Independently verified producer paths protect siblings as well as stores.
    # This runs BEFORE even a control mkdir/lock/token may modify a source tree.
    meta = primitive.inspect_inputs(args)
    primitive._output_paths(args, meta)
    for key, value in meta['paths'].items():
        source = Path(value).resolve()
        if key in ('prices', 'rules') or (key in ('s1', 's2') and source.name.casefold() == key):
            source = source.parent
        if control.is_relative_to(source) or source.is_relative_to(control):
            raise RuntimeRefused('control overlaps requested input source')
    control.mkdir(parents=True, exist_ok=True)
    with primitive.run_lock(control):
        _reserve(control, c)
        return _execute_reserved(c, args, control, policy, meta)


def _execute_reserved(c, args, control, policy, verified_meta):
    """예약된 환경 내에서 scales -> stats -> train_S_cuda 파이프라인을 순차 실행한다."""
    from c_ept import runlog
    from c_ept.workflows import run_s3, run_s1
    from c_ept.runtime import E6Guard
    from c_ept.ept import data, train, scales
    last_probe = {}

    def validated_probe(record):
        validate_e6_probe(record)
        last_probe.clear()
        last_probe.update(record)

    guard = E6Guard(lambda: run_s1.require_no_strata(), validate=validated_probe)
    primitive._e6(guard)
    code = primitive.repository_state()
    meta = primitive.inspect_inputs(args)
    if meta != verified_meta:
        raise RuntimeRefused('independently verified inputs changed before run allocation')
    primitive._output_paths(args, meta)
    protected = [Path(value).resolve() for value in meta['paths'].values()]
    protected += [Path(p).resolve() for p in (r'D:\LOL_Project', r'D:\LOL_Training_20261004', r'D:\LOL_External_20261009')]
    if any(control.is_relative_to(p) or p.is_relative_to(control) for p in protected):
        raise RuntimeRefused('control overlaps independently verified input or frozen tree')
    # _train_stage constructs TrainConfig with its frozen default patience.
    # Refuse drift rather than mutating the scientific function or its defaults.
    cfg = train.TrainConfig(primitive._model_config(c['size']), c['lr'], c['weight_decay'], c['warmup_steps'])
    if type(cfg.patience) is not int or cfg.patience != c['patience']:
        raise RuntimeRefused('inherited trainer patience differs from explicit profile patience=2')
    train.deterministic_cpu(threads=8, seed=args.seed)

    def fingerprint(native_policy):
        return {**primitive._runtime_fingerprint(native_policy),
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}

    runtime = fingerprint(policy)
    if runtime.get('dtype') != 'torch.float32' or runtime.get('threads') != 8 or runtime.get('deterministic_algorithms') is not True:
        raise RuntimeRefused('fixed float32/native8 deterministic runtime required')
    identity = {'args': dict(c), 'operator_config_sha256': operator.config_digest(c),
                'code': code, 'runtime': runtime, 'inputs': meta}

    def check():
        primitive._e6(guard)
        run_s3.check_resume_code(code, primitive.repository_state())
        if fingerprint(prepare_native_runtime()) != runtime or primitive.inspect_inputs(args) != meta:
            raise RuntimeRefused('runtime or independently verified inputs changed during run')

    check()
    run = runlog.new_run('ept_pilot', Path(c['runs_root']))
    with primitive.run_lock(run):
        manifest = {'schema': operator.RUN_SCHEMA, 'stage': 'ept_training_single',
                    'operator_api_version': OPERATOR_API_VERSION, 'code_commit': code['code_commit'],
                    'code_dirty': False, 'created': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'identity': identity, 'run_identity_sha256': primitive._digest(identity),
                    'operator_config_sha256': identity['operator_config_sha256'], 'runtime': runtime,
                    'stages': list(operator.STAGES), 'results': {},
                    'stats_population': 'whole FIT; deterministic FIT/STOP 25% proxy',
                    'limits': {'trainer_cumulative_seconds': 24 * 3600, 'sampled_private_bytes_strictly_below': primitive.MEMORY_CAP,
                               'memory_enforcement': 'epoch/batch boundary samples; not a continuous OS guarantee'}}
        primitive._write(run / 'RUN.json', manifest)
        try:
            primitive._write(run / 'PROCESS.json', _process_record(run, manifest, 'running'))
            # Both durable root records exist before publishing the exact ref.
            # A failure from this point is bound to this run; never guess latest.
            primitive._write(control / 'run.json', {'run': str(run.resolve()), 'code_commit': code['code_commit'],
                             'operator_config_sha256': manifest['operator_config_sha256'],
                             'run_identity_sha256': manifest['run_identity_sha256']})
            ledger = run_s3.Ledger(run)
            bundle = primitive._load_inputs(meta)
            bundle.meta = meta

            def scale_action(bounded):
                path = run / scales.SCALES_NAME
                source = Path(meta['paths']['s3']) / scales.SCALES_NAME
                if source.exists():
                    scales.load_scales(source, expected_provenance=meta['expected_provenance'])
                    from c_ept.data.store_core import write_atomic
                    write_atomic(path, source.read_bytes())
                else:
                    scales.b0_scales(meta['paths']['s3'], out=run, runs_root=args.runs_root, check=check)
                value = scales.load_scales(path, expected_provenance=meta['expected_provenance'])
                return {'values': value, 'provenance': meta['expected_provenance']}, [path]

            factors = primitive._stage(run, manifest, ledger, 'scales', identity, check, scale_action)['values']

            def stats_action(bounded):
                path = run / 'stats/STATS.json'
                if path.exists():
                    raise RuntimeRefused('orphan statistics require a new run')
                fitted = data.fit_stats(bundle.s1, bundle.s2)
                fitted.save(path)
                return {'sha256': fitted.sha256, 'role': 'FIT', 'population': 'whole FIT'}, [path]

            primitive._stage(run, manifest, ledger, 'stats', identity, check, stats_action)
            stats = data.Stats.load(run / 'stats/STATS.json')
            proxy = data.proxy_matches(bundle.s1, args.share, args.seed)

            def train_action(bounded):
                folder = run / 'train/S_cuda'
                if folder.exists():
                    raise RuntimeRefused('training folder exists; no resume/retry')
                fields, outputs = primitive._train_stage(run, args, bundle, stats, factors, proxy, 'S', 'cuda', check)
                history = _verify_final_head(folder, c)
                fields.update(termination_reason=history['termination_reason'], patience=c['patience'],
                              max_epochs=c['max_epochs'], share=c['share'])
                return fields, outputs

            primitive._stage(run, manifest, ledger, 'train_S_cuda', identity, check, train_action)
            check()
            runlog.register(run, manifest, Path(c['catalog_db']))
            primitive._write(run / 'PROCESS.json', _process_record(run, manifest, 'finished', 'completed', 0))
        except BaseException as error:
            primitive._write(run / 'PROCESS.json', _process_record(run, manifest, 'failed', 'failed', primitive._failure_exit_code(error)))
            raise
    print(json.dumps({'run': str(run), 'schema': operator.RUN_SCHEMA, 'control': str(control / 'run.json')}))
    return 0


def main(argv=None):
    """EPT-T002 단일 학습 CLI 진입점."""
    parser = argparse.ArgumentParser(description='EPT-T002 supported single-profile runner; no Resume')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--operator-config-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        policy = prepare_native_runtime()
        c = operator.load_config(operator.config_path(args.config))
        if args.operator_config_sha256 != operator.config_digest(c):
            raise RuntimeRefused('effective operator config digest mismatch')
        return execute(c, args.control, policy)
    except (operator.OperatorError, RuntimeRefused, ValueError, OSError) as error:
        print('EPT training refused: ' + str(error), file=sys.stderr)
        return primitive._failure_exit_code(error)


if __name__ == '__main__':
    raise SystemExit(main())
