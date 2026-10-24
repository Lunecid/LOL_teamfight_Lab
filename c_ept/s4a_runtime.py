"""S4a의 작은 표준 라이브러리 전용 실행 계약.

해석: 새 CLI 프로세스에서 native import 전에 NumPy MKL의 순차 계층을 고정한다.
E6의 탐지 규칙은 원래 생산자가 맡고 여기서는 검토된 기록 형식과 실행 허용만 검사한다.
전역 환경·패키지·S3 정책을 변경하거나 임의 embedding의 안전을 보장하지 않는다.
"""
import os
import sys
from pathlib import Path

RUNTIME_POLICY_VERSION = 1


class RuntimeRefused(ValueError):
    """Unsupported native policy or unavailable/active E6 evidence."""


def prepare_native_runtime():
    """Set/check the process-local supported fresh-CLI native policy.

    해석: 오류 무시 설정을 거부한 뒤 native import 전에 순차 MKL을 선택한다.
    이미 다른 정책으로 native 모듈을 가져온 embedding은 이 계약 밖이다.
    허용된 caller 값만 확인한 뒤 FALSE를 고정하여 sklearn의 setdefault가 오류 은폐를 켜지 못하게 한다.
    """
    override = os.environ.get('KMP_DUPLICATE_LIB_OK', '').strip().casefold()
    if override not in ('', '0', 'false', 'no', 'off'):
        raise RuntimeRefused('KMP_DUPLICATE_LIB_OK 오류 무시 설정을 해제하고 새 프로세스로 실행하세요')
    if any(name in sys.modules for name in ('numpy', 'torch', 'scipy')) and os.environ.get('MKL_THREADING_LAYER') != 'SEQUENTIAL':
        raise RuntimeRefused('native import 이전에 MKL_THREADING_LAYER=SEQUENTIAL을 적용한 새 CLI가 필요합니다')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    return {'policy_version': RUNTIME_POLICY_VERSION, 'mkl_threading_layer': 'SEQUENTIAL',
            'numpy_mkl_parallelism': 'sequential', 'kmp_duplicate_lib_ok': 'disabled',
            'python_executable': str(Path(sys.executable).resolve()),
            'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}


def validate_e6_probe(result):
    """Validate the frozen producer record, not a new process detection policy."""
    fields = {'running','matched','bridges_ignored','scanned','cmdline_unreadable','rule'}
    if type(result) is not dict or set(result) != fields:
        raise RuntimeRefused('E6 확인 불가: 검사 기록 필드가 잘못되었습니다')
    if type(result['running']) is not bool or type(result['matched']) is not list:
        raise RuntimeRefused('E6 확인 불가: running/matched 형식이 잘못되었습니다')
    if not isinstance(result['rule'],str) or not result['rule'].strip():
        raise RuntimeRefused('E6 확인 불가: 탐지 규칙 기록이 없습니다')
    for key in ('bridges_ignored','scanned','cmdline_unreadable'):
        if type(result[key]) is not int or result[key] < 0:
            raise RuntimeRefused('E6 확인 불가: 검사 집계 값이 잘못되었습니다')
    matched = result['matched']
    for entry in matched:
        if (type(entry) is not dict or set(entry) != {'pid','name'}
                or type(entry['pid']) is not int or entry['pid'] < 0
                or not isinstance(entry['name'],str)):
            raise RuntimeRefused('E6 확인 불가: 일치 프로세스 기록이 잘못되었습니다')
    pids = [entry['pid'] for entry in matched]
    if (len(set(pids)) != len(pids)
            or matched != sorted(matched,key=lambda entry:(entry['pid'],entry['name']))
            or result['running'] != bool(matched)):
        raise RuntimeRefused('E6 확인 불가: running/matched 기록이 일관되지 않습니다')
    # A matched name can have an unreadable command line. These counts overlap.
    # An ignored bridge, however, had a readable bridge command and is not matched.
    if (len(matched) + result['bridges_ignored'] > result['scanned']
            or result['cmdline_unreadable'] + result['bridges_ignored'] > result['scanned']):
        raise RuntimeRefused('E6 확인 불가: 프로세스 집계가 scanned와 일치하지 않습니다')
    if result['running']:
        raise RuntimeRefused('E6 거부: Strata 탐지 프로세스가 실행 중입니다')
    return result

