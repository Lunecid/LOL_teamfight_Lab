"""C-EPT 전역 설정 및 핵심 상수 모듈 (c_ept.config).

교전(Engagement) 정의 상수(시공간 연결 임계치, 관측 시점 등), 시계열 윈도우,
소환사의 협곡 맵 규격, 피처 필드 정의, 데이터 저장소 경로 및 스펙 무결성 검증 로직을 제공합니다.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. 교전 단위 정의 및 시계(Clocks) 상수
# -----------------------------------------------------------------------------
# 킬 간 연결 허용 시간 (15초 = 15,000ms): 두 킬의 발생 시각 차이가 15초 이내여야 동일 교전으로 연결됨
G_MS = 15_000

# 킬 간 연결 허용 거리 (2,000 맵 유닛): 두 킬의 유클리드 거리가 2,000 이내여야 함
L_U = 2_000.0

# 교전 직전 기준 시점 tau의 리드 타임 (15초): 첫 킬(f) 15초 전(tau = f - 15초)을 상태 관측 기준점으로 삼음
TAU_LEAD_MS = 15_000

# 교전 종료 윈도우 하한선 (20초): 종료 시점 delta = max(마지막 킬 + 1ms, 첫 킬 + 20초)
DELTA_AFTER_F_MS = 20_000

# 유효 교전 시간 범위: 게임 시작 2분(120초) 이후부터 게임 종료 30초 전까지의 교전만 모델링 대상으로 포함
SCOPE_FIRST_MS = 120_000
END_CUT_MS = 30_000

# -----------------------------------------------------------------------------
# 2. 프레임 시계, 발생 빈(Bin), 및 전환(Conversion) 윈도우
# -----------------------------------------------------------------------------
# 교전 발생 예측 지평창 (180초 = 3분): 현재 시점 t0 이후 3분 내 교전 발생 여부를 예측
H_OCC_MS = 180_000

# 다중 지평 평가 단계 (60초, 120초, 180초): 발생 확률 예측 성능을 지평별로 분할 평가
EVAL_H_MS = (60_000, 120_000, 180_000)

# 이산화 시간 빈(Bin) 간격 (10초): 180초 지평을 10초 단위 총 18개 빈으로 분할
BIN_MS = 10_000
N_BINS = 18

# 교전 후 오브젝트 전환 윈도우 (60초): 교전 종료(delta) 후 60초 내에 타워/용/바론 등을 획득했는지 측정
CONV_MS = 60_000
assert N_BINS * BIN_MS == H_OCC_MS == EVAL_H_MS[-1] and all(h % BIN_MS == 0 for h in EVAL_H_MS)

# -----------------------------------------------------------------------------
# 3. 소환사의 협곡 맵 규격, 마크, 이벤트 토큰 설정
# -----------------------------------------------------------------------------
# 소환사의 협곡 가로/세로 맵 변 길이 (14,875 유닛):
# (x, y) -> (W - x, W - y)로 180도 점대칭 회전 시 블루팀과 레드팀 진영이 완벽히 뒤바뀜
W = 14_875.0

# 첫 킬 매치업 표지 총 개수: 50종(챔피언 간 킬) + 2종(처형) = 52종
N_MARKS = 52

# 최근 이벤트 관측 윈도우 (교전 직전 2분 = 120초): 트랜스포머 인코더에 들어갈 최근 사건 시퀀스
RECENT_MS = 120_000

# 과거 이벤트 관측 윈도우 (교전 직전 10분 = 600초): 장기 이력 누적 통계용
OLD_MS = 600_000

# 시퀀스 모델 최대 토큰 수 (최대 128개 최근 이벤트, 최대 128개 동시 참여 엣지)
MAX_EVENTS = 128
MAX_EDGES = 128

# -----------------------------------------------------------------------------
# 4. 데이터셋 범주 및 필드 이름 목록
# -----------------------------------------------------------------------------
# 데이터셋 대상 LoL 패치 버전 목록
PATCHES = ('15.18', '15.19', '15.20', '15.21', '15.22', '16.13')

# 데이터 분할(Split) 역할군 (학습, 검증, 테스트, 캘리브레이션 등)
ROLES = ('FIT', 'UNUSED', 'STOP', 'DEV', 'CAL', 'AUDIT', 'TEST', 'EXTERNAL')

# 주요 중립 몬스터 및 구조물 오브젝트 종류
O_KINDS = ('dragon', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower', 'inhibitor', 'plate')

# 타임라인 내 수집 대상 이벤트 타입
EVENT_TYPES = (
    'CHAMPION_KILL', 'ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED',
    'ITEM_PURCHASED', 'LEVEL_UP', 'WARD_PLACED', 'WARD_KILL'
)

# Riot Match-V5 타임라인 participantFrames 내의 25개 상세 챔피언 스탯 키 (알파벳 정렬)
CHAMPION_STATS_KEYS = (
    'abilityHaste', 'abilityPower', 'armor', 'armorPen', 'armorPenPercent', 'attackDamage', 'attackSpeed',
    'bonusArmorPenPercent', 'bonusMagicPenPercent', 'ccReduction', 'cooldownReduction', 'health', 'healthMax',
    'healthRegen', 'lifesteal', 'magicPen', 'magicPenPercent', 'magicResist', 'movementSpeed', 'omnivamp',
    'physicalVamp', 'power', 'powerMax', 'powerRegen', 'spellVamp'
)

# 1분 단위 타임라인 프레임 행에 저장되는 33개 필드 (기본 8개 + 챔피언 스탯 25개)
FRAME_FIELDS = (
    'x', 'y', 'level', 'xp', 'currentGold', 'totalGold', 'minionsKilled', 'jungleMinionsKilled',
    *('cs_' + k for k in CHAMPION_STATS_KEYS)
)
assert len(CHAMPION_STATS_KEYS) == 25 and len(FRAME_FIELDS) == 33

_SIZE_CLASS_KEYS = ('T', 'S', 'ASYM', 'P')

# -----------------------------------------------------------------------------
# 5. 파일 시스템 경로 관리
# -----------------------------------------------------------------------------
def _path(env: str, default: str) -> Path:
    """환경 변수가 설정되어 있으면 해당 경로를, 없으면 기본 경로를 Path 객체로 반환합니다."""
    return Path(os.environ.get(env) or default)


CACHE_SETS = {
    '15.18': 'ext_kr_15_18_22', '15.19': 'ext_kr_15_18_22', '15.20': 'ext_kr_15_18_22',
    '15.21': 'ext_kr_15_18_22', '15.22': 'ext_kr_15_18_22', '16.13': 'ext_na1_16_13'
}
RAW_REGION = {
    '15.18': 'kr', '15.19': 'kr', '15.20': 'kr',
    '15.21': 'kr', '15.22': 'kr', '16.13': 'na1'
}
assert set(CACHE_SETS) == set(RAW_REGION) == set(PATCHES)

RAW_ROOT = _path('C_EPT_RAW_ROOT', r'D:\LOL_Data\patches')
CACHE_ROOT = _path('C_EPT_CACHE_ROOT', r'D:\LOL_Data\patches')
RUNS_ROOT = _path('C_EPT_RUNS_ROOT', r'D:\LOL_Data\runs')
CATALOG_DB = _path('C_EPT_CATALOG_DB', r'D:\LOL_Data\catalog\runs.sqlite')
LEGACY_ROOT = _path('C_EPT_LEGACY_ROOT', r'C:\Users\todtj\문서\LOL_Teamfight\worktrees\engagement-state-value')
SPLIT_DIR = _path('C_EPT_SPLIT_DIR', r'C:\Users\todtj\.codex\visualizations\2026\10\15\01a13f40-d38f-76e3-9108-1b668280290d\holdout_r3')
SPEC_PATH = _path('C_EPT_SPEC_PATH', r'D:\LOL_Data\runs\20261019_constant_drift_fit_1\SPEC_C_DEFINITION.json')


def constants() -> dict:
    """현재 모듈에 정의된 모든 대문자 상수를 정렬된 JSON 직렬화 가능 딕셔너리로 추출합니다.
    실험 재현성을 위해 실행 매니페스트에 설정 해시를 남길 때 사용됩니다.
    """
    out = {
        name: (list(v) if isinstance(v, tuple) else v)
        for name, v in globals().items()
        if name.isupper() and not name.startswith('_') and isinstance(v, (int, float, str, tuple))
    }
    return dict(sorted(out.items()))


# -----------------------------------------------------------------------------
# 6. 설계 스펙 무결성 검증 (Spec Verification)
# -----------------------------------------------------------------------------
class SpecMismatch(RuntimeError):
    """외부 결정 기록(SPEC JSON)과 본 코드 상수가 불일치할 때 발생하는 예외"""


def check_spec(path: Path | str = SPEC_PATH) -> str:
    """SPEC_C_DEFINITION.json 결정 기록과 현재 코드의 핵심 상수(G_ms, L_u, tau_lead_ms 등)를 대조합니다.
    불일치가 없으면 해당 결정 기록의 SHA-256 해시를 반환합니다.
    """
    raw = Path(path).read_bytes()
    spec = json.loads(raw.decode('utf-8'))
    op = spec.get('operating_point') if isinstance(spec, dict) else None
    if not isinstance(op, dict):
        raise SpecMismatch(f'{path}: no operating_point object')
    bad = [
        f'operating_point.{key} is {op.get(key, "<absent>")!r}, code has {want!r}'
        for key, want in (('G_ms', G_MS), ('L_u', L_U), ('tau_lead_ms', TAU_LEAD_MS))
        if op.get(key) != want
    ]
    if 'presence_gate' not in op or op['presence_gate'] is not None:
        bad.append(f'operating_point.presence_gate is {op.get("presence_gate", "<absent>")!r}, code expects null')
    classes = op.get('size_classes')
    missing = [k for k in _SIZE_CLASS_KEYS if not isinstance(classes, dict) or k not in classes]
    if missing:
        bad.append(f'operating_point.size_classes lacks {missing}')
    if bad:
        raise SpecMismatch(f'{path} differs from c_ept.config: ' + '; '.join(bad))
    return hashlib.sha256(raw).hexdigest()


# -----------------------------------------------------------------------------
# 7. 데이터 분할(Split) 모듈 동적 로더
# -----------------------------------------------------------------------------
_SPLIT = None


def split_module():
    """데이터 분할 모듈(split.py)을 안전하게 동적 임포트하여 반환합니다.
    바이트코드(.pyc) 생성을 방지하여 읽기 전용 저장소 정책을 준수합니다.
    """
    global _SPLIT
    if _SPLIT is None:
        d = str(SPLIT_DIR)
        if d not in sys.path:
            sys.path.insert(0, d)
        importlib.invalidate_caches()
        keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            mod = importlib.import_module('split')
        finally:
            sys.dont_write_bytecode = keep
        if not os.path.samefile(mod.__file__, SPLIT_DIR / 'split.py'):
            raise ImportError(f'module "split" came from {mod.__file__}, not from SPLIT_DIR')
        _SPLIT = mod
    return _SPLIT
