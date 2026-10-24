"""Constants, paths and name lists of the C engagement-process model (c_ept).

Every number below is fixed by the approved design; ``check_spec`` compares the ones the decision record states.
Times are milliseconds, distances are map units (``_U``).


방법에서의 위치. 이 모듈은 S1의 교전 단위와 S2의 입력 보기가 공유하는 상수·열 순서·경로를 제공한다.
교전은 시간·거리 조건으로 연결한 킬 묶음이고, 분 프레임은 경기 상태의 정기 기록이다(설계서 §2–4).
상수 목록과 결정 기록을 대조하는 절차는 S1–S2 계획 Task 1에 해당한다.
식 목록. 이 모듈은 계산 모형을 정의하지 않으며 새 식 번호를 두지 않는다.
입력은 환경 변수와 결정 기록이고, 출력은 설정 사전·결정 원본의 해시·분할 모듈이다.
공통 가정. 시간 상수는 ms, 거리는 지도 단위이며, 각 창의 실제 절단은 해당 계산 모듈이 맡는다.
모듈 수준 assert는 구간·열 수·패치 키의 일관성을 검사하며 Python 최적화 실행에서는 생략될 수 있다.
결정 기록 검사는 지정한 운영점 필드와 규모 분류 키만 확인하며, 분류의 값 전체를 검증하지 않는다(Task 1).
회전 설명의 map_side는 코드 상수 W의 지도 변 길이이며, 전역 기호 W(승자)와 구분한다(S3 판정 R57).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 2-4 and 11 (constants and name lists).
# Ref: D:\LOL_Data\runs\20261019_constant_drift_fit_1\SPEC_C_DEFINITION.json, key operating_point (decision record: G, L, tau lead,
#      presence gate, size classes).
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

# ---- engagement unit and its clocks ---------------------------------------------------------------------------------
G_MS = 15_000                # link gap: kills i ~ j iff |t_i - t_j| <= G_MS and the distance is <= L_U (single linkage)
L_U = 2_000.0                # link distance in map units; the test is dx*dx + dy*dy <= L_U*L_U (boundary included)
TAU_LEAD_MS = 15_000         # reference time tau = f - TAU_LEAD_MS for the first kill f
DELTA_AFTER_F_MS = 20_000    # delta = max(last kill + 1, f + DELTA_AFTER_F_MS)
SCOPE_FIRST_MS = 120_000     # an engagement is in scope iff SCOPE_FIRST_MS <= f <= T - END_CUT_MS
END_CUT_MS = 30_000

# ---- frame-clock rows, occurrence bins, conversion window -----------------------------------------------------------
H_OCC_MS = 180_000                       # occurrence window (t0, min(t0 + H_OCC_MS, T - END_CUT_MS)]
EVAL_H_MS = (60_000, 120_000, 180_000)   # evaluation horizons: the first 6 / 12 / 18 bins
BIN_MS = 10_000
N_BINS = 18
CONV_MS = 60_000                         # conversion window (delta, min(delta + CONV_MS, T)]
assert N_BINS * BIN_MS == H_OCC_MS == EVAL_H_MS[-1] and all(h % BIN_MS == 0 for h in EVAL_H_MS)

# ---- map, marks, event tokens ---------------------------------------------------------------------------------------
W = 14_875.0          # map side; the 180-degree rotation (map_x, map_y) -> (map_side - map_x, map_side - map_y) swaps the two teams
N_MARKS = 52
RECENT_MS = 120_000   # recent events: (ref - RECENT_MS, ref], at most MAX_EVENTS of them
OLD_MS = 600_000      # old events: (ref - OLD_MS, ref - RECENT_MS]
MAX_EVENTS = 128      # S2 gate S2G-1, 2026-10-21; was 80
MAX_EDGES = 128       # co-participation edges

# ---- name lists -----------------------------------------------------------------------------------------------------
PATCHES = ('15.18', '15.19', '15.20', '15.21', '15.22', '16.13')
ROLES = ('FIT', 'UNUSED', 'STOP', 'DEV', 'CAL', 'AUDIT', 'TEST', 'EXTERNAL')
O_KINDS = ('dragon', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower', 'inhibitor', 'plate')
EVENT_TYPES = ('CHAMPION_KILL', 'ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED',
               'ITEM_PURCHASED', 'LEVEL_UP', 'WARD_PLACED', 'WARD_KILL')
# raw Match-V5 timeline participantFrames[*].championStats, in file (alphabetical) order
CHAMPION_STATS_KEYS = ('abilityHaste', 'abilityPower', 'armor', 'armorPen', 'armorPenPercent', 'attackDamage', 'attackSpeed',
                       'bonusArmorPenPercent', 'bonusMagicPenPercent', 'ccReduction', 'cooldownReduction', 'health', 'healthMax',
                       'healthRegen', 'lifesteal', 'magicPen', 'magicPenPercent', 'magicResist', 'movementSpeed', 'omnivamp',
                       'physicalVamp', 'power', 'powerMax', 'powerRegen', 'spellVamp')
FRAME_FIELDS = ('x', 'y', 'level', 'xp', 'currentGold', 'totalGold', 'minionsKilled', 'jungleMinionsKilled',
                *('cs_' + k for k in CHAMPION_STATS_KEYS))
assert len(CHAMPION_STATS_KEYS) == 25 and len(FRAME_FIELDS) == 33
_SIZE_CLASS_KEYS = ('T', 'S', 'ASYM', 'P')   # keys the decision record must keep under operating_point.size_classes


# ---- paths (each can be moved with an environment variable) ----------------------------------------------------------
def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or default)


# <root>/<patch>/raw/<RAW_REGION[patch]>/{timeline,detail}/<key>.json   and
# <root>/<patch>/cache/<CACHE_SETS[patch]>/<key>.{meta.json,events.json,npz}
CACHE_SETS = {'15.18': 'ext_kr_15_18_22', '15.19': 'ext_kr_15_18_22', '15.20': 'ext_kr_15_18_22', '15.21': 'ext_kr_15_18_22',
              '15.22': 'ext_kr_15_18_22', '16.13': 'ext_na1_16_13'}   # cache bundle folder of each patch
RAW_REGION = {'15.18': 'kr', '15.19': 'kr', '15.20': 'kr', '15.21': 'kr', '15.22': 'kr', '16.13': 'na1'}   # raw region folder of each patch
assert set(CACHE_SETS) == set(RAW_REGION) == set(PATCHES)
RAW_ROOT = _path('C_EPT_RAW_ROOT', r'D:\LOL_Data\patches')
CACHE_ROOT = _path('C_EPT_CACHE_ROOT', r'D:\LOL_Data\patches')
RUNS_ROOT = _path('C_EPT_RUNS_ROOT', r'D:\LOL_Data\runs')                    # <YYYYMMDD>_c_ept_<stage>_<n> folders
CATALOG_DB = _path('C_EPT_CATALOG_DB', r'D:\LOL_Data\catalog\runs.sqlite')   # run index (guideline C7)
LEGACY_ROOT = _path('C_EPT_LEGACY_ROOT', r'C:\Users\todtj\문서\LOL_Teamfight\worktrees\engagement-state-value')   # read only
SPLIT_DIR = _path('C_EPT_SPLIT_DIR', r'C:\Users\todtj\.codex\visualizations\2026\10\15\01a13f40-d38f-76e3-9108-1b668280290d\holdout_r3')
# decision record, read only
SPEC_PATH = _path('C_EPT_SPEC_PATH', r'D:\LOL_Data\runs\20261019_constant_drift_fit_1\SPEC_C_DEFINITION.json')


def constants() -> dict:
    """The method constants (numbers, strings, name lists) as a JSON-ready dict, sorted by name; paths are left out.

    S1–S2 계획 Task 1의 설정 해시 입력으로, 공개 대문자 이름의 숫자·문자열·튜플을 모으고 튜플은 목록으로 바꾼다.
    경로와 사전형 설정은 이 선택에 포함되지 않는다.
    """
    out = {name: (list(v) if isinstance(v, tuple) else v) for name, v in globals().items()
           if name.isupper() and not name.startswith('_') and isinstance(v, (int, float, str, tuple))}
    return dict(sorted(out.items()))


# ---- decision record --------------------------------------------------------------------------------------------------
class SpecMismatch(RuntimeError):
    """The decision record states a constant that differs from this module.

    S1–S2 계획 Task 1의 운영점 대조 실패를 구분하는 예외이다.
    """


def check_spec(path: Path | str = SPEC_PATH) -> str:
    """Compare the decision record's operating point with the constants here; return the record's SHA-256.

    Compared: operating_point.G_ms, L_u, tau_lead_ms, presence_gate (must be null) and the size-class keys T/S/ASYM/P.
    The record grows with later decisions, so its hash is not pinned: callers write it into the run manifest.


    S1–S2 계획 Task 1에 따라 운영점의 지정 필드와 규모 분류 키를 대조한 뒤 원본 바이트의 해시를 반환한다.
    이 검사는 결정 원본 해시를 외부의 기대값에 고정하는 검사가 아니며, 규모 분류 키가 더 있거나 그 값이 달라도 거부하지 않는다.
    """
    raw = Path(path).read_bytes()
    spec = json.loads(raw.decode('utf-8'))
    op = spec.get('operating_point') if isinstance(spec, dict) else None
    if not isinstance(op, dict):
        raise SpecMismatch(f'{path}: no operating_point object')
    bad = [f'operating_point.{key} is {op.get(key, "<absent>")!r}, code has {want!r}'
           for key, want in (('G_ms', G_MS), ('L_u', L_U), ('tau_lead_ms', TAU_LEAD_MS)) if op.get(key) != want]
    if 'presence_gate' not in op or op['presence_gate'] is not None:
        bad.append(f'operating_point.presence_gate is {op.get("presence_gate", "<absent>")!r}, code expects null')
    classes = op.get('size_classes')
    missing = [k for k in _SIZE_CLASS_KEYS if not isinstance(classes, dict) or k not in classes]
    if missing:
        bad.append(f'operating_point.size_classes lacks {missing}')
    if bad:
        raise SpecMismatch(f'{path} differs from c_ept.config: ' + '; '.join(bad))
    return hashlib.sha256(raw).hexdigest()


# ---- cohort split module ----------------------------------------------------------------------------------------------
_SPLIT = None


def split_module():
    """Import the cohort split module once and return it. Call only its ``bucket()`` and ``roles()``.

    SPLIT_DIR goes on sys.path (once, in front) so that ``import split`` finds it; the module's own text is never read here. The import
    runs with ``sys.dont_write_bytecode`` on (put back afterwards, also when the import fails): the folder is read only and a .pyc of
    the module would hold its salt literal, as the module does.


    S1–S2 판정 R33의 읽기 전용 불러오기를 적용하고 최초 성공 때 모듈의 실제 파일 경로를 확인한다.
    성공 후에는 캐시한 모듈을 반환하며, 분할 정책 본문을 여기서 문서화하지 않는다.
    """
    global _SPLIT
    if _SPLIT is None:
        d = str(SPLIT_DIR)
        if d not in sys.path:
            sys.path.insert(0, d)
        importlib.invalidate_caches()
        keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True       # as legacy._import_modules does for LEGACY
        try:
            mod = importlib.import_module('split')
        finally:
            sys.dont_write_bytecode = keep
        if not os.path.samefile(mod.__file__, SPLIT_DIR / 'split.py'):
            raise ImportError(f'module "split" came from {mod.__file__}, not from SPLIT_DIR')
        _SPLIT = mod
    return _SPLIT
