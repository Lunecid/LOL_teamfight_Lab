r"""[모듈 개요 및 하네스 위치]
이 모듈은 EPT 신경망 학습 과정에서 에폭별 학습 및 검증 지표를 누적 집계하고,
불변 스냅샷 파일(epoch_metrics.json, epoch_metrics.csv)로 보존하는 진단 엔진이다 (S4a 과제 5; v6 스키마).

A. 하네스 및 파이프라인 내 위치:
   - 외부 의존성 없는 표준 라이브러리(stdlib) 전용 구현으로, 대용량 텐서/데이터 로딩 없이 가볍고 안전하게 작동한다.
   - 단일 진실 공급원(Single Source of Truth): 최신 완료된 `epoch_XXXX/head.json`에 의해
     파일 크기(bytes) 및 SHA-256 해시가 검증 결합된다.
   - 에폭 표기는 1-based(사용자 친화적), 에폭 인덱스는 0-based(코드 루프)로 명확히 분리 관리.

B. 지표 정의 및 평가 규약:
   1) fit_batch_mean_loss: 학습 모드(training mode)에서의 FIT 배치 손실 산술 평균 (학습 진행 진단용, 공식 평가 아님).
   2) stop_loss: 평가 모드(evaluation mode)에서 풀링된 점수 행/경기 가중/규모 층 손실 (판정 R-S4a-16).
      배치 평균의 평균이 아니며, 학습 손실과 직접 비교할 수 없는 정밀 조기 종료 판정 기준.
   3) lr_last_step: 해당 에폭의 마지막 옵티마이저 스텝에서 적용된 학습률.

C. 주요 함수 및 안전성 보장:
   - bounded_read: 최대 읽기 바이트(MAX_BYTES, 16MB) 제한을 두어 메모리 초과 공격 및 파손 파일 방지.
   - read_snapshot / write_snapshot: 불변 스냅샷 입출력 및 무결성 검증.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

SCHEMA = 'epoch-metrics-v1'
HEAD_SCHEMA = 'ept_train_checkpoint_v6'
FILES = ('epoch_metrics.json', 'epoch_metrics.csv')
MAX_BYTES = 16 * 1024 * 1024
COLUMNS = ('epoch', 'epoch_index', 'fit_batch_mean_loss', 'stop_loss', 'lr_last_step')
DEFINITIONS = {
    'fit_batch_mean_loss': 'arithmetic mean of FIT batch losses in training mode; diagnostic, not evaluation',
    'stop_loss': 'existing evaluate_stop pooled scored-row/match/size-stratum loss (R-S4a-16), evaluation mode; not a mean of batch means',
    'lr_last_step': 'learning rate used by the last optimizer step of this epoch',
    'epoch': '1-based completed epoch label',
    'epoch_index': '0-based loop index; best_epoch_index also 0-based',
}
MONITOR = {'name': 'STOP', 'mode': 'min', 'improvement': 'strict', 'min_delta': 0.0}


def _require(ok):
    """조건이 참이 아닐 경우 체크포인트 에러를 발생시킨다."""
    if not ok:
        raise RuntimeError('checkpoint epoch metrics missing or invalid; not guessing an older checkpoint')


def _integer(value, low=0):
    """값이 특정 하한선 이상의 정수형인지 검증한다."""
    return type(value) is int and value >= low


def _number(value):
    """값이 유한한 실수 또는 정수인지 검증한다."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def json_bytes(doc):
    """메타데이터 딕셔너리를 정렬된 JSON 포맷 UTF-8 바이트로 직렬화한다."""
    return (json.dumps(doc, sort_keys=True, indent=1, allow_nan=False) + '\n').encode('utf-8')


def bounded_read(path):
    """최대 메모리 버짓(MAX_BYTES) 내에서 파일 바이트를 안전하게 읽는다."""
    try:
        with Path(path).open('rb') as handle:
            raw = handle.read(MAX_BYTES + 1)
        _require(len(raw) <= MAX_BYTES)
        return raw
    except OSError as err:
        raise RuntimeError('checkpoint metadata cannot be read; not guessing an older checkpoint') from err


def csv_bytes(rows):
    """행 리스트를 규격 CSV 포맷 UTF-8 바이트로 직렬화한다."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def validate(doc, head):
    """모델 페이로드 없이 스칼라 지표로부터 조기 종료 및 최적 에폭 메타데이터를 검증한다."""
    try:
        n, horizon, seed, identity = head['epoch'], head['epochs'], head['seed'], head['identity']
        patience = identity['patience']
        _require(_integer(n, 1) and n <= 9999 and _integer(horizon, 1) and n <= horizon)
        _require(_integer(seed) and _integer(patience, 1))
        _require(type(identity['epochs']) is int and identity['epochs'] == horizon)
        _require(type(identity['seed']) is int and identity['seed'] == seed)
        _require(doc['schema'] == SCHEMA and doc['identity'] == identity and doc['seed'] == seed)
        _require(doc['epoch_count'] == n and type(doc['epoch_count']) is int)
        _require(doc['definitions'] == DEFINITIONS and doc['monitor'] == MONITOR and doc['patience'] == patience)
        rows = doc['rows']
        _require(isinstance(rows, list) and len(rows) == n)
        best, best_index, bad = math.inf, -1, 0
        for index, row in enumerate(rows):
            _require(isinstance(row, dict) and set(row) == set(COLUMNS))
            _require(type(row['epoch']) is int and row['epoch'] == index + 1)
            _require(type(row['epoch_index']) is int and row['epoch_index'] == index)
            _require(all(_number(row[key]) for key in COLUMNS[2:]) and row['lr_last_step'] >= 0)
            if row['stop_loss'] < best:
                best, best_index, bad = row['stop_loss'], index, 0
            else:
                bad += 1
            _require(index == n - 1 or bad < patience)
        early = bad >= patience
        done = early or n == horizon
        reason = 'early_stopping' if early else 'max_epochs' if done else 'in_progress'
        for key, value in {'best_epoch_index': best_index, 'best_epoch': best_index + 1,
                           'bad_epochs': bad, 'done': done, 'stopped_early': early,
                           'termination_reason': reason}.items():
            _require(type(doc[key]) is type(value) and doc[key] == value)
        _require(type(head['best_epoch']) is int and head['best_epoch'] == best_index)
        _require(type(head['done']) is bool and head['done'] == done)
        _require(type(head['stopped_early']) is bool and head['stopped_early'] == early)
        _require(head['best']['folder'] == f'epoch_{best_index + 1:04d}')
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)
    return doc


def snapshot(loop, head):
    """루프 상태나 난수 스트림을 변경하지 않고 누적된 스칼라 지표 딕셔너리를 추출한다."""
    try:
        n = head['epoch']
        _require(type(loop['epoch']) is int and loop['epoch'] == n)
        _require(len(loop['fit_batch_mean_losses']) == len(loop['stop_losses']) == len(loop['steps']) == n)
        _require(all(_integer(step, 1) for step in loop['steps']))
        _require(isinstance(head['timing']['steps'], list) and len(head['timing']['steps']) == n)
        _require(all(_integer(step, 1) for step in head['timing']['steps']))
        _require(loop['steps'] == head['timing']['steps'])
        _require(sum(loop['steps']) == len(loop['lrs']))
        _require(all(_number(lr) and lr >= 0 for lr in loop['lrs']))
        _require(_number(loop['best_value']) and loop['best_value'] == min(loop['stop_losses']))
        rows, offset = [], 0
        for index, count in enumerate(loop['steps']):
            offset += count
            rows.append(dict(zip(COLUMNS, (index + 1, index, loop['fit_batch_mean_losses'][index],
                                           loop['stop_losses'][index], loop['lrs'][offset - 1]))))
        early, done = loop['stopped_early'], head['done']
        doc = {'schema': SCHEMA, 'identity': head['identity'], 'seed': head['seed'], 'epoch_count': n,
               'definitions': DEFINITIONS, 'monitor': MONITOR, 'patience': head['identity']['patience'],
               'best_epoch_index': loop['best_epoch'], 'best_epoch': loop['best_epoch'] + 1,
               'bad_epochs': loop['bad'], 'done': done, 'stopped_early': early,
               'termination_reason': 'early_stopping' if early else 'max_epochs' if done else 'in_progress', 'rows': rows}
        return validate(doc, head)
    except (KeyError, TypeError, IndexError):
        _require(False)


def encode(loop, head):
    """스냅샷 결과를 JSON 및 CSV 바이트 딕셔너리로 인코딩하여 반환한다."""
    doc = snapshot(loop, head)
    return dict(zip(FILES, (json_bytes(doc), csv_bytes(doc['rows']))))


def read_snapshot(folder, head):
    """고정된 JSON/CSV 스냅샷 파일들을 읽어 크기 및 SHA-256 해시를 검증하고 반환한다."""
    try:
        binding = head['epoch_metrics']
        _require(head['schema'] == HEAD_SCHEMA and binding['schema'] == SCHEMA)
        _require(set(binding['files']) == set(FILES))
        raw = {}
        for name in FILES:
            entry = binding['files'][name]
            _require(set(entry) == {'bytes', 'sha256'} and _integer(entry['bytes'], 1) and entry['bytes'] <= MAX_BYTES)
            _require(isinstance(entry['sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None)
            raw[name] = bounded_read(Path(folder) / name)
            _require(len(raw[name]) == entry['bytes'] and hashlib.sha256(raw[name]).hexdigest() == entry['sha256'])
        doc = validate(json.loads(raw[FILES[0]]), head)
        _require(raw[FILES[1]] == csv_bytes(doc['rows']))
        return doc
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)
