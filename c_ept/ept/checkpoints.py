r"""[모듈 개요 및 하네스 위치]
이 모듈은 EPT 신경망 학습 과정에서 가중치 텐서(best.pt, last.pt)와 메타데이터(head.json),
진단 지표(epoch_metrics.json, epoch_metrics.csv)를 안전하게 디스크에 직렬화하고 복구하는
헤드 후행(Head-Last) 체크포인트 IO 엔진이다 (S4a 과제 5; v5 및 v6 규격 지원).

A. 하네스 및 파이프라인 내 위치:
   - train.py 루프와 training_operator에서 호출되며, 전원 장애나 비정상 종료 시에도 안전한 재개(Resume)를 보장한다.
   - 단일 진실 공급원(Single Source of Truth): 가장 최근 완료된 에폭 폴더의 head.json이 커밋 표지 역할을 수행한다.
   - 텐서 파일(best.pt, last.pt)과 스칼라 진단 지표가 온전히 디스크에 기록되고 검증된 후에만
     마지막으로 head.json을 기록하여 불완전한 상태(torn folder)의 체크포인트가 커밋되는 것을 방지한다.

B. 스키마 규격 및 하위 호환성:
   - v5 스키마 ('ept_train_checkpoint_v5'): 과거 체크포인트 호환 규격.
   - v6 스키마 ('ept_train_checkpoint_v6'): 불변 epoch_metrics.json/csv 스냅샷을 필수 포함하며,
     크기(bytes) 및 SHA-256 해시를 head.json에 강제 바인딩.
   - v6 게시 후 이전 v5로의 다운그레이드는 엄격히 차단된다.

C. 주요 함수 및 안전성 보장:
   - write_atomic: 임시 파일(.tmp) 기록 -> fsync -> os.replace 원자적 대체로 단일 파일 무결성 보장.
   - latest_metadata: 대용량 텐서 페이로드를 읽지 않고 가벼운 head.json 및 스칼라 메타데이터만 신속 조회.
   - latest_checkpoint: 메타데이터 및 SHA-256 해시 검증 통과 후 텐서 바이트를 메모리에 로드.
   - save_checkpoint: 가중치 작성 -> 지표 작성 -> head.json 커밋의 헤드 후행(head-last) 보증.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

from c_ept.ept import epoch_metrics

SCHEMA = 'ept_train_checkpoint_v5'  # 과거 v5 헤드 및 호출 회계 스키마
METRICS_SCHEMA = epoch_metrics.HEAD_SCHEMA
COMPONENT_FIELDS = ('transfer_seconds', 'forward_seconds', 'backward_seconds', 'optimizer_seconds')
FILES = ('last.pt', 'best.pt')


def sha256(payload: bytes) -> str:
    """파일 검증에 쓰는 바이트 SHA-256."""
    return hashlib.sha256(payload).hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    """임시 파일 동기화 후 대체 방식으로 단일 파일을 원자적으로 디스크에 기록한다.

    매개변수:
    - path: 최종 대상 파일 경로.
    - payload: 저장할 바이트 데이터.

    동작 및 안전성 보장:
    - 'path.tmp'에 데이터를 기록한 뒤 flush 및 os.fsync를 호출하여 파일 내용을 동기화한다. 디렉터리 fsync나 전원 장애 후 내구성까지 보장하지는 않는다.
    - 동기화 완료 후 os.replace를 수행하므로 대상 경로는 불완전한 중간 상태 없이 온전한 파일로 대체된다.
    - 본 함수는 단일 파일 쓰기의 원자성만 보장하며, 복수 파일 트랜잭션을 구성하지 않는다."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def torch_bytes(obj) -> bytes:
    """메모리 객체만 직렬화하며 체크포인트 파일은 읽지 않는다."""
    import torch
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def json_bytes(doc: dict) -> bytes:
    """딕셔너리를 정렬된 JSON 포맷(indent=1) UTF-8 바이트로 직렬화한다."""
    return (json.dumps(doc, indent=1, sort_keys=True) + '\n').encode('utf-8')


def read_verified(path: Path, entry: dict, label: str) -> bytes:
    """head.json에 기록된 크기(bytes) 및 SHA-256 해시와 대조 검증하여 파일 바이트를 반환한다."""
    try:
        payload = path.read_bytes()
    except OSError as err:
        raise RuntimeError(f'checkpoint {label}: {path.name} is listed in head.json but cannot be read ({err.__class__.__name__}); not guessing an older one') from None
    if len(payload) != entry['bytes'] or sha256(payload) != entry['sha256']:
        raise RuntimeError(f'checkpoint {label}: {path.name} does not match the size and SHA-256 in head.json; not guessing an older checkpoint')
    return payload


def latest_metadata(out_dir: Path):
    """가장 최근 커밋된 에폭 폴더로부터 (folder, head, history) 메타데이터를 조회하여 반환한다.

    - history는 원격 측정 마커/파일이 없는 과거 v5 규격에서는 None이다.
    - v6 규격은 고정된 이름의 JSON/CSV 쌍을 필수로 요구하며 .pt 접근 전에 모든 스칼라 메타데이터를 사전 검증한다.
    - head.json이 누락된 손상 폴더(torn folder)는 건너뛰며, 손상된 완료 헤드는 즉시 에러로 처리한다.
    - 본 API는 텐서 페이로드의 실제 디스크 가용성까지는 검증하지 않는다.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    folders = sorted((p for p in out_dir.glob('epoch_[0-9][0-9][0-9][0-9]') if p.is_dir()), reverse=True)
    for folder in folders:
        head = folder / 'head.json'
        if not head.is_file():
            continue
        try:
            meta = json.loads(epoch_metrics.bounded_read(head))
            best = meta['best']
            ok = (meta['schema'] in (SCHEMA, METRICS_SCHEMA) and type(meta['epoch']) is int and folder.name == f'epoch_{int(meta["epoch"]):04d}' and 'last.pt' in meta['files'] and set(meta['files']) <= set(FILES)
                  and set(best) == {'folder', 'sha256', 'bytes'} and re.fullmatch(r'epoch_\d{4}', best['folder']) is not None)
        except (OSError, ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            raise RuntimeError(f'checkpoint {folder.name}: head.json is unreadable or not of the schema {SCHEMA}; not guessing an older checkpoint')
        timing = meta.get('timing', {})
        if any(key not in timing or not isinstance(timing[key], list) or len(timing[key]) != int(meta['epoch'])
               or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in timing[key]) for key in COMPONENT_FIELDS):
            raise RuntimeError('checkpoint telemetry schema/components are missing or invalid')
        if meta['schema'] == METRICS_SCHEMA:
            history = epoch_metrics.read_snapshot(folder, meta)
        else:
            if 'epoch_metrics' in meta or any((folder / name).exists() for name in epoch_metrics.FILES):
                raise RuntimeError('checkpoint epoch metrics schema mismatch; not guessing an older checkpoint')
            # 한 번 v6를 게시한 실행은 이후 v5 헤드로 다운그레이드할 수 없음
            scanned_bytes = 0
            for prior in folders:
                if prior.name >= folder.name or not (prior / 'head.json').is_file():
                    continue
                if any((prior / name).exists() for name in epoch_metrics.FILES):
                    raise RuntimeError('checkpoint v6 to v5 downgrade; not guessing an older checkpoint')
                raw_head = epoch_metrics.bounded_read(prior / 'head.json')
                scanned_bytes += len(raw_head)
                if scanned_bytes > epoch_metrics.MAX_BYTES:
                    raise RuntimeError('checkpoint legacy provenance scan exceeds metadata budget')
                try:
                    old_head = json.loads(raw_head)
                    legacy = old_head['schema'] == SCHEMA and 'epoch_metrics' not in old_head
                except (ValueError, TypeError, KeyError):
                    legacy = False
                if not legacy:
                    raise RuntimeError('checkpoint v6 to v5 downgrade or invalid prior head; not guessing an older checkpoint')
            history = None  # 과거 v5에서는 지표 이력이 제공되지 않음 (임의 합성 금지)
        return folder, meta, history
    return None


def latest_checkpoint(out_dir: Path):
    """텐서 바이트 로드 전 메타데이터와 SHA-256을 선행 검증하여 최신 체크포인트 텐서를 반환한다."""
    found = latest_metadata(Path(out_dir))
    if found is None:
        return None
    folder, meta, _ = found
    best = meta['best']
    raw = {name: read_verified(folder / name, entry, folder.name) for name, entry in meta['files'].items()}
    raw['best.pt'] = raw['best.pt'] if best['folder'] == folder.name else read_verified(Path(out_dir) / best['folder'] / 'best.pt', best, folder.name)
    return folder, meta, raw['last.pt'], raw['best.pt']


def save_checkpoint(out_dir: Path, epoch_done: int, files: dict, meta: dict, *, write=None) -> tuple:
    """가중치 파일 작성 후 완료 표지(head.json)를 기록하는 헤드 후행 방식으로 체크포인트를 저장한다.

    매개변수:
    - out_dir: 체크포인트 상위 디렉터리.
    - epoch_done: 완료된 에폭 번호(int).
    - files: 파일명과 바이트 데이터 매핑 (last.pt, best.pt).
    - meta: head.json에 기록될 메타데이터 딕셔너리.
    - write: 파일 쓰기 함수(기본값: write_atomic).

    반환값:
    - (folder, meta, head_sha256) 튜플.

    예외 및 경계 조건:
    - 대상 에폭 폴더에 head.json 경로가 이미 존재하면(내용의 유효성을 여기서 검사하지 않음) FileExistsError를 발생시켜 덮어쓰기를 방지한다.
    - 이전 쓰기 중단으로 head.json이 없는 torn 폴더는 내부 파일을 덮어쓰며 안전하게 재사용된다.
    - best가 변경되지 않은 경우 files에서 best.pt를 생략할 수 있다. 쓰기 콜백의 예외는 전파된다."""
    write = write_atomic if write is None else write
    folder = out_dir / f'epoch_{epoch_done:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'head.json').exists():
        raise FileExistsError(f'{folder.name} already holds a complete checkpoint; nothing is overwritten')
    if meta['schema'] == METRICS_SCHEMA:
        if not set(epoch_metrics.FILES) <= set(files):
            raise RuntimeError('checkpoint epoch metrics files required for v6')
        meta = {**meta, 'epoch_metrics': {'schema': epoch_metrics.SCHEMA, 'files': {
            name: {'sha256': sha256(files[name]), 'bytes': len(files[name])} for name in epoch_metrics.FILES}}}
    for name in (*FILES, *(epoch_metrics.FILES if meta['schema'] == METRICS_SCHEMA else ())):
        if name in files:
            write(folder / name, files[name])
    meta = {**meta, 'files': {name: {'sha256': sha256(files[name]), 'bytes': len(files[name])} for name in FILES if name in files}}
    if meta['schema'] == METRICS_SCHEMA:
        epoch_metrics.read_snapshot(folder, meta)
    payload = json_bytes(meta)
    write(folder / 'head.json', payload)
    return folder, meta, sha256(payload)
