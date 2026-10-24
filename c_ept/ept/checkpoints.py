"""EPT v5 체크포인트 직렬화 및 원자적 IO 경계 모듈.

설계 규격 및 특성:
- SCHEMA = 'ept_train_checkpoint_v5' 규격을 준수한다.
- 상태 텐서 파일(last.pt, best.pt)을 먼저 기록하고, 완료 표지인 head.json을 마지막에 작성한다.
- 개별 파일은 임시 경로 flush 및 fsync 후 os.replace로 원자 교체되지만, 다중 파일 디렉터리 전체가 단일 트랜잭션으로 커밋되는 것은 아니다.
- 완료 표지가 누락된 미완료 폴더(torn write)는 재기록을 허용하며, 손상된 체크포인트는 과거 상태로 후퇴하지 않고 즉시 예외를 발생시킨다."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

SCHEMA = 'ept_train_checkpoint_v5'
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
    return (json.dumps(doc, indent=1, sort_keys=True) + '\n').encode('utf-8')


def read_verified(path: Path, entry: dict, label: str) -> bytes:
    """The bytes of ``path``, checked against the size and SHA-256 that a head.json lists (RuntimeError, naming the checkpoint ``label``, otherwise)."""
    try:
        payload = path.read_bytes()
    except OSError as err:
        raise RuntimeError(f'checkpoint {label}: {path.name} is listed in head.json but cannot be read ({err.__class__.__name__}); not guessing an older one') from None
    if len(payload) != entry['bytes'] or sha256(payload) != entry['sha256']:
        raise RuntimeError(f'checkpoint {label}: {path.name} does not match the size and SHA-256 in head.json; not guessing an older checkpoint')
    return payload


def latest_checkpoint(out_dir: Path):
    """출력 디렉터리에서 최신 완료 체크포인트의 메타데이터와 검증된 바이트를 읽는다. 텐서를 역직렬화하지 않는다.

    매개변수:
    - out_dir: 에폭 폴더들이 위치한 기본 디렉터리 경로.

    반환값:
    - (folder, meta, last_bytes, best_bytes) 튜플. 완료된 체크포인트가 없으면 None.

    검증 및 손상 격리 정책:
    - epoch_XXXX 형식의 폴더를 역순 탐색하며, head.json이 없는 미완료(torn) 폴더는 건너뛴다.
    - head.json의 스키마, 에폭 일치 여부, 타이밍 텔레메트리 유한성/길이, 파일 해시(SHA-256) 및 바이트 크기를 엄격히 검증한다.
    - 검증 실패나 손상 감지 시 이전 과거 에폭으로 임의 후퇴하지 않고 RuntimeError를 발생시킨다.
    - best.pt는 head가 지정한 과거 에폭 폴더에 있을 수 있으며 해당 바이트도 검증한다."""
    if not out_dir.is_dir():
        return None
    for folder in sorted((p for p in out_dir.glob('epoch_[0-9][0-9][0-9][0-9]') if p.is_dir()), reverse=True):
        head = folder / 'head.json'
        if not head.is_file():
            continue
        try:
            meta = json.loads(head.read_text(encoding='utf-8'))
            best = meta['best']
            ok = (meta['schema'] == SCHEMA and folder.name == f'epoch_{int(meta["epoch"]):04d}' and 'last.pt' in meta['files'] and set(meta['files']) <= set(FILES)
                  and set(best) == {'folder', 'sha256', 'bytes'} and re.fullmatch(r'epoch_\d{4}', best['folder']) is not None)
        except (OSError, ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            raise RuntimeError(f'checkpoint {folder.name}: head.json is unreadable or not of the schema {SCHEMA}; not guessing an older checkpoint')
        timing = meta.get('timing', {})
        if any(key not in timing or not isinstance(timing[key], list) or len(timing[key]) != int(meta['epoch'])
               or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in timing[key]) for key in COMPONENT_FIELDS):
            raise RuntimeError('checkpoint telemetry schema/components are missing or invalid')
        raw = {name: read_verified(folder / name, entry, folder.name) for name, entry in meta['files'].items()}
        raw['best.pt'] = raw['best.pt'] if best['folder'] == folder.name else read_verified(out_dir / best['folder'] / 'best.pt', best, folder.name)
        return folder, meta, raw['last.pt'], raw['best.pt']
    return None


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
    for name in FILES:
        if name in files:
            write(folder / name, files[name])
    meta = {**meta, 'files': {name: {'sha256': sha256(files[name]), 'bytes': len(files[name])} for name in FILES if name in files}}
    payload = json_bytes(meta)
    write(folder / 'head.json', payload)
    return folder, meta, sha256(payload)
