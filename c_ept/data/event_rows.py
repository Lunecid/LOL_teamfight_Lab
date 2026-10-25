"""C-EPT 이벤트 시계(Event Clock) 행 생성 모듈 (c_ept.data.event_rows).

교전(Engagement) 발생 시점에서의 모델 예측 스냅샷 행들을 생성합니다.
- M3 (KIND_M3 = 1): 교전 첫 킬 시각 f 시점에서의 예측 (교전 결과 킬 성과 K, 오브젝트 성과 O 예측)
- M4 (KIND_M4 = 2): 교전 종료 시각 delta 시점에서의 예측 (교전 후 60초 내 오브젝트 전환 성과 예측)
- 각 스냅샷 시점(ref) 바로 직전에 기록된 정규 타임라인 프레임(frame_k)과 프레임 나이(age_ms = ref - t0)를 매핑합니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from c_ept.data.frame_rows import FrameRows
from c_ept.data.reader import MatchRecord
from c_ept.domain.units import UnitTable

# 이벤트 시계 관측 종류:
# 1 (KIND_M3): 첫 킬 시각 f에서의 관측 (교전 결과 예측)
# 2 (KIND_M4): 교전 종료 시각 delta에서의 관측 (오브젝트 전환 예측)
KIND_M3, KIND_M4 = 1, 2


@dataclass(frozen=True, eq=False)
class EventRows:
    """단일 매치에서 추출된 이벤트 시계 스냅샷 행 테이블입니다.
    - unit: 소속 교전 단위 인덱스 (UnitTable 기준)
    - kind: 관측 종류 (KIND_M3 또는 KIND_M4)
    - ref: 관측 기준 시각 (ms: M3는 첫 킬 f, M4는 교전 종료 delta)
    - frame_k: ref 직전의 마지막 정규 프레임 번호 (k)
    - t0: 해당 정규 프레임의 실제 시각 (ms)
    - age_ms: 프레임 나이 (ref - t0: 프레임 기록 후 이벤트 발생까지 경과한 시간)
    - frame_row: 해당 프레임이 FrameRows 테이블에 존재하는 경우 그 행 번호 (-1이면 미포함)
    """
    unit: np.ndarray           # 교전 단위 번호 (int32)
    kind: np.ndarray           # 관측 종류 (int8: KIND_M3 또는 KIND_M4)
    ref: np.ndarray            # 기준 시각 (밀리초, ms, int64)
    frame_k: np.ndarray        # 직전 정규 프레임 번호 (int16)
    t0: np.ndarray             # 프레임 시각 (밀리초, ms, int64)
    age_ms: np.ndarray         # 프레임 나이 (ref - t0 ms, int64)
    frame_row: np.ndarray      # 프레임 시계 행 인덱스 (int32, 없으면 -1)


def event_rows(rec: MatchRecord, u: UnitTable, rows: FrameRows) -> EventRows:
    """유효 교전(in_scope)들에 대해 M3 및 M4 이벤트 시계 스냅샷 행들을 생성합니다.

    - M3 행: 모든 유효 교전의 첫 킬 시각 f를 기준 시각으로 생성
    - M4 행: delta < T_ms (경기 종료 전)인 교전에 대해 종료 시각 delta를 기준으로 생성
    - 각 기준 시각(ref)보다 엄격히 이전(ts < ref)에 위치한 가장 최신 정규 프레임(frame_k)을 이진 탐색으로 조회
    """
    rk = np.asarray(rows.k, dtype=np.int64)
    if rk.size > 1 and np.any(rk[1:] <= rk[:-1]):
        raise ValueError('frame-clock rows must be in frame order: k must increase strictly')
    ts = np.asarray(rec.frame_ts, dtype=np.int64)

    # 유효 범위(in_scope) 내의 교전 단위들만 선별
    unit = np.flatnonzero(u.in_scope)

    # 각 단위마다 M3(첫 킬 f)와 M4(종료 delta) 후보 스냅샷 생성
    c_unit = np.repeat(unit, 2)
    c_kind = np.tile(np.array([KIND_M3, KIND_M4], dtype=np.int8), unit.size)
    c_ref = np.stack([u.f[unit], u.delta[unit]], axis=1).astype(np.int64).ravel()

    # 기준 시각 ref보다 엄격히 앞선 마지막 정규 프레임 탐색 (searchsorted - 1)
    c_k = np.searchsorted(ts, c_ref, side='left') - 1

    # 유효성 검사: ref 이전에 프레임이 최소 1개 이상 존재해야 하고, M4는 경기 종료 전에 끝나야 함
    keep = (c_k >= 0) & ((c_kind == KIND_M3) | (c_ref < rec.T_ms))
    ref, k = c_ref[keep], c_k[keep]
    t0 = ts[k]

    # 해당 프레임이 FrameRows 테이블에 정의된 행인지 확인하고 인덱스 매핑
    pos = np.searchsorted(rk, k, side='left')
    frame_row = np.where(np.append(rk, -1)[pos] == k, pos, -1)

    return EventRows(
        unit=c_unit[keep].astype(np.int32),
        kind=c_kind[keep],
        ref=ref,
        frame_k=k.astype(np.int16),
        t0=t0,
        age_ms=ref - t0,
        frame_row=frame_row.astype(np.int32)
    )
