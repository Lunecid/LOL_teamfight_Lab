"""Hash-pinned adapter for the game-rule modules of the new lineage (LEGACY, read only).

The rules (respawn, objective timers, item replay, champion attributes) are older, tested code that is not rewritten here. Every file
they use is pinned by SHA-256 in ``legacy_pins.json``; ``verify`` and ``load`` refuse to continue when one of them differs, until
the author decides to re-pin (``write_pins``).


방법에서의 위치. 이 모듈은 새 계보의 게임 규칙을 재사용하는 해시 고정 어댑터이다(설계서 §11 항목 10).
LEGACY는 외부의 규칙 작업공간이며, 반환하는 객체는 부활·오브젝트 시계·아이템·챔피언 속성 모듈이다.
입력은 규칙 루트와 저장된 파일별 핀이고, 출력은 검증한 핀 또는 불러온 모듈 묶음이다.
식 목록. 해시·목록 검사·불러오기만 수행하므로 별도 수학식을 정의하지 않는다.
공통 가정. 핀 JSON의 구조는 올바르다고 가정하며, 검사 범위는 expected_files가 열거한 파일이다.
파일 집합과 바이트 해시의 일치는 확인하지만 현재 외부 Git HEAD를 핀의 head와 대조하지 않는다(Task 1).
불러온 모듈은 프로세스 안에서 재사용하므로 디스크 재검사와 모듈 재적재는 서로 다른 절차이다.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 11, item 10 (game-rule code reuse behind a hash-pinned adapter).
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from c_ept import config as C

PINS_PATH = Path(__file__).with_name('legacy_pins.json')
_MODULES = ('event_survival', 'objective_timers', 'item_state', 'champion_attributes')   # what load() returns
_GAMEPLAY = ('__init__', 'event_survival', 'respawn_rules', 'objective_timers', 'item_state', 'champion_attributes')
_RULE_FILES = ('objective_rules', 'item_effect_flags', 'map_anchors', 'item_rules_15x', 'item_rules_16x')

_LOADED: SimpleNamespace | None = None


class LegacyPinError(RuntimeError):
    """A pinned LEGACY file is missing or differs from legacy_pins.json (or was imported from somewhere else).

    S1–S2 계획 Task 1의 규칙 핀·불러오기 출처 검사 실패를 구분한다.
    """


def expected_files() -> list[str]:
    """Paths (relative to LEGACY_ROOT, forward slashes, sorted) that the pin list must cover exactly.

    S1–S2 계획 Task 1에서 정한 규칙 파일과 패치별 표의 상대 경로를 정렬하여 핀의 검사 범위를 정한다.
    """
    files = [f'gameplay/{n}.py' for n in _GAMEPLAY]
    files += [f'config/game_rules/{n}.json' for n in _RULE_FILES]
    files += [f'config/game_rules/datadragon_v2/{kind}_{p}.json' for kind in ('champions', 'items') for p in C.PATCHES]
    return sorted(files)


def _sha256(path: Path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pins() -> dict:
    """The parsed pin file: {'head': LEGACY git commit the pins were taken from, 'files': {relative path: SHA-256}}.

    S1–S2 계획 Task 1의 핀 기록을 읽으며, 이 함수만으로 파일 해시의 일치를 검사하지는 않는다.
    """
    return json.loads(PINS_PATH.read_text(encoding='utf-8'))


def pins_sha256() -> str:
    """SHA-256 of the canonical JSON of the pins, so it does not change with the checkout's line endings.

    S1–S2 계획 Task 1의 실행 기록에 넣을 핀 내용의 정렬된 JSON 해시를 만든다.
    """
    return hashlib.sha256(json.dumps(pins(), sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def verify(root: Path | None = None) -> dict:
    """Check every pinned file under ``root`` (default: the current ``config.LEGACY_ROOT``); return the pin dict.

    Raises LegacyPinError naming each file that is missing or differs.


    S1–S2 계획 Task 1의 파일 목록과 바이트 해시를 검사하며, 저장된 head는 현재 Git HEAD와 대조하지 않는다.
    """
    root = Path(C.LEGACY_ROOT if root is None else root)
    pinned = pins()
    expected = expected_files()
    if sorted(pinned['files']) != expected:
        gone, extra = sorted(set(expected) - set(pinned['files'])), sorted(set(pinned['files']) - set(expected))
        raise LegacyPinError(f'{PINS_PATH.name} must list exactly the expected files (not listed: {gone}; unexpected: {extra})')
    problems = []
    for rel in expected:
        p = root / rel
        if not p.is_file():
            problems.append(f'{rel}: missing')
        elif _sha256(p) != pinned['files'][rel]:
            problems.append(f'{rel}: differs from its pin')
    if problems:
        raise LegacyPinError(f'LEGACY files at {root} do not match {PINS_PATH.name}: ' + '; '.join(problems)
                             + '. Re-pin (write_pins) only by the author\'s decision.')
    return pinned


def load() -> SimpleNamespace:
    """Verify the pins, then import and return the four LEGACY modules (``head`` is the pinned commit).

    The pins are checked on every call; the modules are imported once per process.


    S1–S2 계획 Task 1의 규칙 재사용 경로로, 매번 핀 검사를 거친 뒤 프로세스별로 캐시한 모듈 묶음을 반환한다.
    """
    global _LOADED
    pinned = verify()
    if _LOADED is None:
        _LOADED = SimpleNamespace(**_import_modules(Path(C.LEGACY_ROOT)), head=pinned['head'])
    return _LOADED


def _import_modules(root: Path) -> dict:
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    importlib.invalidate_caches()
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # LEGACY is read only: no .pyc written into it
    try:
        mods = {name: importlib.import_module(f'gameplay.{name}') for name in _MODULES}
    finally:
        sys.dont_write_bytecode = keep
    gameplay = (root / 'gameplay').resolve()
    for name, mod in mods.items():
        if not Path(mod.__file__).resolve().is_relative_to(gameplay):
            raise LegacyPinError(f'gameplay.{name} was imported from {mod.__file__}, not from the pinned {gameplay}')
    return mods


def write_pins(root: Path | None = None) -> dict:
    """Pin the files of ``root`` as they are now (and its git HEAD) into legacy_pins.json. Author's decision only.

    S1–S2 계획 Task 1의 저자 결정에 따른 핀 갱신 기능이며, 주석 보강 작업에서는 호출하지 않는다.
    """
    root = Path(C.LEGACY_ROOT if root is None else root)
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    pinned = {'head': head, 'files': {rel: _sha256(root / rel) for rel in expected_files()}}
    PINS_PATH.write_text(json.dumps(pinned, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    return pinned
