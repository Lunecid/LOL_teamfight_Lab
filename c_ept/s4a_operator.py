"""Agent-free S4a operator. Read-only preflight; runner retains execution authority.

Uses shared fresh-CLI startup/E6 and the agreed RUN/PROCESS metadata contract.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from c_ept.s4a_runtime import RuntimeRefused, prepare_native_runtime, validate_e6_probe

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = re.compile(r'^\d{8}_c_ept_s4a_\d+$')

class OperatorError(ValueError):
    pass

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result

def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite JSON number')
    return number


def read_json(path):
    """Read bounded metadata only; never deserialize Python or model objects."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size > 4 * 1024 * 1024:
        raise OperatorError(f'메타데이터 없음/크기 초과: {p}')
    try:
        data = json.loads(p.read_text(encoding='utf-8-sig'),
                          object_pairs_hook=unique_object,
                          parse_float=finite_float,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, UnicodeError, ValueError) as exc:
        raise OperatorError(f'손상된 JSON 메타데이터: {p.name}') from exc
    if not isinstance(data, dict):
        raise OperatorError(f'JSON 객체 필요: {p.name}')
    return data

def load_config(path, *, require_inputs=True):
    c = read_json(path)
    expected = {'schema','python','s1','s2','s3','runs_root','catalog_db','sizes',
                'devices','epochs','share','dev_sample','task7_integration_verified'}
    if set(c) != expected or c['schema'] != 's4a-operator-v1':
        raise OperatorError('설정 키 또는 schema 불일치')
    for k in ('python','s1','s2','s3','runs_root','catalog_db'):
        if not isinstance(c[k], str) or not c[k] or not Path(c[k]).is_absolute():
            raise OperatorError(f'{k}: 절대 경로를 설정하세요')
    if c['sizes'] != ['S','M'] or c['devices'] != ['cpu','cuda']:
        raise OperatorError('승인 범위: sizes S/M, devices cpu/cuda')
    if type(c['epochs']) is not int or c['epochs'] != 2:
        raise OperatorError('승인 범위: epochs=2')
    if type(c['share']) not in (float,int) or c['share'] != .25:
        raise OperatorError('승인 범위: share=0.25')
    if type(c['dev_sample']) is not int or c['dev_sample'] != 200:
        raise OperatorError('승인 범위: dev_sample=200')
    if type(c['task7_integration_verified']) is not bool:
        raise OperatorError('task7_integration_verified는 bool 필요')
    runs = Path(c['runs_root']).resolve()
    if runs != Path(r'D:\LOL_Data\runs').resolve():
        raise OperatorError('runs_root는 D:\\LOL_Data\\runs로 제한됩니다')
    catalog = Path(c['catalog_db']).resolve()
    if not catalog.is_relative_to(runs.parent):
        raise OperatorError('catalog_db는 D:\\LOL_Data 아래에 있어야 합니다')
    if require_inputs:
        for k in ('s1','s2','s3'):
            if not Path(c[k]).is_dir():
                raise OperatorError(f'{k} 입력 경로 없음: {c[k]}')
        if not Path(c['python']).is_file():
            raise OperatorError('Python 경로 없음. 기존 Anaconda 경로를 확인하세요')
        if Path(c['python']).resolve() != Path(sys.executable).resolve():
            raise OperatorError('설정 Python과 실행 Python이 다릅니다. -Python 경로를 맞추세요')
    return c

def runtime_check():
    spec = read_json(ROOT/'environment/s4a-runtime.json')
    problems = []
    if platform.python_version() != spec['python_version']:
        problems.append(f"Python {spec['python_version']} 필요 (현재 {platform.python_version()})")
    for name, expected in spec['packages'].items():
        try:
            got = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            got = '미설치'
        if got != expected:
            problems.append(f'{name}: 필요 {expected}, 현재 {got}')
    if problems:
        raise OperatorError('; '.join(problems)+'; 자동 설치하지 않습니다')

def code_check():
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    def git(*args):
        p = subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,
                           text=True,env=env,check=False)
        if p.returncode:
            raise OperatorError('Git 코드 상태 확인 실패')
        return p.stdout.strip()
    commit = git('rev-parse','HEAD')
    if git('status','--porcelain','--untracked-files=normal'):
        raise OperatorError('코드가 깨끗하지 않습니다. 검토된 커밋에서 실행하세요')
    return commit

def e6_check():
    # Reuse project policy; None/unknown is a refusal, never just a warning.
    try:
        from c_ept import runlog
        result = runlog.strata_check()
    except Exception as exc:
        raise OperatorError('E6 확인 불가: runlog.strata_check 인터페이스 통합 필요') from exc
    try:
        result = validate_e6_probe(result)
    except RuntimeRefused as exc:
        # Display only safe name/PID fields, never command lines. This projection
        # is diagnostic; the shared validator alone decides refusal/permission.
        matched = result.get('matched') if type(result) is dict else None
        safe = [{'pid':item['pid'],'name':item['name']} for item in matched
                if type(item) is dict and type(item.get('pid')) is int
                and isinstance(item.get('name'),str)] if type(matched) is list else []
        detail = '; '+json.dumps(safe,ensure_ascii=False) if safe else ''
        raise OperatorError(str(exc)+detail) from exc
    print('E6 진단: '+json.dumps({key:result[key] for key in
          ('scanned','bridges_ignored','cmdline_unreadable')},ensure_ascii=False)+
          '; 일치 프로세스가 탐지되지 않았습니다. 이 검사는 절대 부재 증명이 아닙니다.')
    return {'running':False}

def gpu_metadata():
    # NVIDIA query is metadata only; no tensor creation, CUDA kernels or training.
    try:
        result = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version',
                                 '--format=csv,noheader'],capture_output=True,text=True,
                                timeout=15,check=False)
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise OperatorError('GPU 메타데이터 확인 실패: NVIDIA 드라이버와 nvidia-smi 확인') from exc
    if result.returncode or not result.stdout.strip():
        raise OperatorError('요청된 CUDA 장치의 메타데이터를 확인할 수 없습니다')
    return result.stdout.strip().splitlines()

def preflight(c):
    runtime_check()
    commit = code_check()
    print('GPU 메타데이터 (계산 검증 아님): '+json.dumps(gpu_metadata(),ensure_ascii=False))
    # Fixed metadata filenames only. Full source verification belongs to runner.
    metadata = {}
    for key in ('s1','s2'):
        metadata[key] = read_json(Path(c[key])/'STORE.json')
    metadata['s3_run'] = read_json(Path(c['s3'])/'RUN.json')
    metadata['s3_stages'] = read_json(Path(c['s3'])/'STAGES.json')
    e6_check()
    if not c['task7_integration_verified']:
        raise OperatorError('후보 패키지: Task7 통합 검증 전에는 실행할 수 없습니다')
    return commit

def run_path(value, c):
    if not value:
        raise OperatorError('Resume/Status에는 -Run 경로를 명시하세요')
    p = Path(value).resolve()
    if p.parent != Path(c['runs_root']).resolve() or not RUN_NAME.fullmatch(p.name):
        raise OperatorError('Run은 승인된 runs_root 바로 아래 YYYYMMDD_c_ept_s4a_N 형식이어야 합니다')
    if not p.is_dir():
        raise OperatorError('지정 Run이 없습니다')
    return p

def canonical_absolute(value, label):
    if not isinstance(value,str) or not Path(value).is_absolute():
        raise OperatorError(label+': 절대 경로가 필요합니다')
    try:
        resolved = Path(value).resolve()
    except (OSError,ValueError,RuntimeError) as exc:
        raise OperatorError(label+': 경로를 해석할 수 없습니다') from exc
    if str(resolved) != value:
        raise OperatorError(label+': canonical 경로가 필요합니다')
    return resolved


def sha_field(value, label, *, nullable=False, length=64):
    if value is None and nullable:
        return value
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}',value):
        raise OperatorError(label+': 해시 형식이 잘못되었습니다')
    return value


def validate_run_metadata(record, path):
    required = {'schema','stage','operator_api_version','identity','run_identity_sha256',
                'code_commit','operator_config_sha256','code_dirty','runtime'}
    if (not required.issubset(record) or record.get('schema') != 's4a_run_v1' or record.get('stage') != 's4a'
            or type(record.get('operator_api_version')) is not int
            or record['operator_api_version'] != 1):
        raise OperatorError('RUN schema/stage/API 불일치')
    identity = record.get('identity')
    keys = {'args','operator_config_sha256','code','runtime','inputs'}
    if type(identity) is not dict or set(identity) != keys:
        raise OperatorError('RUN identity 구조 불일치')
    for key in ('args','code','runtime','inputs'):
        if type(identity[key]) is not dict:
            raise OperatorError('RUN identity.'+key+' 형식 불일치')
    code = identity['code']
    if set(code) != {'code_commit','code_dirty','repo_root'} or code['code_dirty'] is not False:
        raise OperatorError('RUN code identity 구조/청결 기록 불일치')
    canonical_absolute(code['repo_root'],'RUN repo_root')
    sha_field(code['code_commit'],'RUN code_commit',length=40)
    sha_field(identity['operator_config_sha256'],'RUN operator_config_sha256',nullable=True)
    # Must match Task7 _digest: ensure_ascii=True is deliberate, including Unicode.
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),
                                       ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    if (record.get('run_identity_sha256') != digest
            or record.get('code_commit') != code['code_commit']
            or record.get('operator_config_sha256') != identity['operator_config_sha256']
            or record.get('code_dirty',False) is not False
            or ('runtime' in record and record['runtime'] != identity['runtime'])):
        raise OperatorError('RUN identity digest/code/config/runtime 결합 불일치')
    if canonical_absolute(identity['args'].get('runs_root'),'RUN runs_root') != path.parent:
        raise OperatorError('RUN 출력 루트와 조회 경로 불일치')
    return identity, digest


def validate_process_metadata(record, run, identity, digest):
    fields = {'schema','operator_api_version','run_path','run_identity_sha256',
              'code_commit','operator_config_sha256','pid','create_time','state',
              'outcome','exit_code','updated_at'}
    if (set(record) != fields or record.get('schema') != 's4a_process_v1'
            or type(record.get('operator_api_version')) is not int
            or record['operator_api_version'] != 1):
        raise OperatorError('PROCESS schema/API/필드 불일치')
    if (canonical_absolute(record['run_path'],'PROCESS run_path') != run
            or record['run_identity_sha256'] != digest
            or record['code_commit'] != identity['code']['code_commit']
            or record['operator_config_sha256'] != identity['operator_config_sha256']):
        raise OperatorError('PROCESS/RUN 경로·identity·code·config 결합 불일치')
    if type(record['pid']) is not int or record['pid'] <= 0:
        raise OperatorError('PROCESS owner PID 형식 오류')
    if type(record['create_time']) is not float or not math.isfinite(record['create_time']) or record['create_time'] <= 0:
        raise OperatorError('PROCESS creation time 형식 오류')
    try:
        stamp = dt.datetime.fromisoformat(record['updated_at'])
        if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
            raise ValueError('UTC required')
    except (TypeError,ValueError) as exc:
        raise OperatorError('PROCESS updated_at UTC timestamp 오류') from exc
    state, outcome, code = record['state'], record['outcome'], record['exit_code']
    valid = ((state == 'running' and outcome is None and code is None)
             or (state == 'finished' and outcome == 'completed' and type(code) is int and code == 0)
             or (state == 'failed' and outcome == 'failed'
                 and (code is None or (type(code) is int and code != 0))))
    if not valid:
        raise OperatorError('PROCESS state/outcome/exit_code 조합 오류')


def observe_process(pid):
    # No native numerical imports, command-line inspection or run-lock acquisition.
    try:
        import psutil
    except ImportError:
        return {'kind':'unavailable'}
    try:
        owner = psutil.Process(pid)
        if not owner.is_running():
            return {'kind':'missing'}
        return {'kind':'observed','create_time':owner.create_time()}
    except psutil.NoSuchProcess:
        return {'kind':'missing'}
    except psutil.AccessDenied:
        return {'kind':'access_denied'}
    except (OSError,psutil.Error,ValueError,OverflowError):
        return {'kind':'unavailable'}


def status(path):
    path = Path(path).resolve()
    manifest = read_json(path/'RUN.json')
    identity, digest = validate_run_metadata(manifest,path)
    artifacts = {'RUN.json':str(path/'RUN.json')}
    result = {'state':'unknown','message':'PROCESS 소유권 기록 없음: 실행 상태 미확인',
              'run_identity_sha256':digest,'artifacts':artifacts}
    for name in ('STAGES.json','COST.json'):
        p = path/name
        if not p.exists():
            continue
        metadata = read_json(p)
        artifacts[name] = str(p)
        if name == 'COST.json':
            if metadata.get('schema') != 's4a_cost_v1' or metadata.get('run_identity_sha256') != digest:
                raise OperatorError('COST schema/identity 불일치')
        else:
            stages = metadata.get('stages')
            if metadata.get('schema') != 's3_stages_v1' or type(stages) is not dict or type(metadata.get('calls')) is not list:
                raise OperatorError('STAGES schema/형식 불일치')
            if any(type(v) is not dict or type(v.get('complete')) is not bool for v in stages.values()):
                raise OperatorError('STAGES complete 형식 불일치')
            result['progress'] = {'recorded_stages':len(stages),
                                  'complete_stages':sum(v['complete'] for v in stages.values()),
                                  'liveness_inferred':False}
    p = path/'PROCESS.json'
    if not p.exists():
        return result
    record = read_json(p)
    validate_process_metadata(record,path,identity,digest)
    artifacts['PROCESS.json'] = str(p)
    if record['state'] == 'finished':
        result.update(state='recorded_completion', recorded_exit_code=record['exit_code'],
                      message='파이프라인 완료 기록: OS 종료 관측이나 연구 결과 수락을 뜻하지 않습니다')
    elif record['state'] == 'failed':
        result.update(state='recorded_failure', recorded_exit_code=record['exit_code'],
                      message='파이프라인 실패 기록: OS 종료를 별도로 관측한 결과는 아닙니다')
    else:
        evidence = observe_process(record['pid'])
        result['process_evidence'] = evidence['kind']
        if evidence['kind'] == 'observed' and evidence.get('create_time') == record['create_time']:
            result.update(state='running', message='조회 시점의 PID·생성시간·실행 identity 일치')
        elif evidence['kind'] == 'missing':
            result.update(state='interrupted_unconfirmed', message='기록된 프로세스 없음: 중단 원인과 종료 코드는 미확인')
        else:
            result['message'] = 'PID 재사용 또는 프로세스 조회 불가: 실행 상태 미확인'
    return result


def config_digest(c):
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def runner_args(c, resume=None):
    args=[]
    for k in ('s1','s2','s3','runs_root','catalog_db','epochs','share','dev_sample'):
        args.extend(['--'+k.replace('_','-'), str(c[k])])
    args.extend(['--sizes',*c['sizes'],'--devices',*c['devices']])
    if resume is not None:
        args.extend(['--resume', str(resume)])
    return args

def main(argv=None):
    p=argparse.ArgumentParser(description='S4a 수동 실행: 기본 Check (읽기 전용)')
    p.add_argument('--action',choices=['Check','Start','Resume','Status'],default='Check')
    p.add_argument('--config',type=Path,default=ROOT/'configs/s4a.local.json')
    p.add_argument('--run',type=Path)
    a=p.parse_args(argv)
    try:
        policy = prepare_native_runtime() if a.action != 'Status' else None
        c=load_config(a.config.resolve(), require_inputs=a.action != 'Status')
        # Echo configuration, never entire metadata containing private identifiers.
        print(json.dumps({'action':a.action,'config':str(a.config.resolve()),
                          'selection':c,'config_sha256':config_digest(c),'runtime_policy':policy},ensure_ascii=False,indent=2))
        if a.action=='Status':
            print(json.dumps(status(run_path(a.run,c)),ensure_ascii=False,indent=2))
            return 0
        if a.action!='Resume' and a.run is not None:
            raise OperatorError('--run은 Resume/Status에서만 사용하세요')
        resume=run_path(a.run,c) if a.action=='Resume' else None
        commit=preflight(c)
        print('메타데이터 사전 점검 통과. 전체 원천 검증은 실행기가 수행합니다. commit='+commit)
        if a.action=='Check':
            return 0
        # Only explicit Start/Resume can import runner or invoke its main.
        runner=importlib.import_module('c_ept.run_s4a')
        # Accepted Task7 must enforce this identity, not just display it.
        if getattr(runner,'OPERATOR_API_VERSION',None) != 1:
            raise OperatorError('Task7 operator API v1 통합 확인 필요')
        result=runner.main([*runner_args(c,resume),'--operator-config-sha256',config_digest(c)])
        return 0 if result is None else int(result)
    except (OperatorError,RuntimeRefused,OSError) as exc:
        print('실행 거부: '+str(exc),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
