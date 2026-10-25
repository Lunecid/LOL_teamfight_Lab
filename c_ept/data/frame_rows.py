"""C-EPT 프레임 시계(Frame Clock) 행 모듈 (c_ept.data.frame_rows).

경기 타임라인의 분 단위 정규 프레임(Frame Clock)을 기준으로 미래 교전 발생 여부를 예측하는
관측 행(FrameRows)과 발생(Onsets) 테이블을 생성합니다.
- 2:00 이후(k >= 2)부터 경기 종료 30초 전까지의 프레임을 질문 시점(t0)으로 채택
- 향후 3분(180초) 발생 지평창을 10초 단위(18개 빈)로 쪼개어 유효 관측 노출(exposure_ms) 계산
- 각 프레임 창 내에서 첫 킬이 일어난 교전들을 (row, unit) 발생 쌍(Onsets)으로 매핑
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from c_ept import config as C
from c_ept.data.reader import MatchRecord
from c_ept.domain.units import UnitTable

# 첫 번째 분석 대상 정규 프레임 인덱스 (k = 2: 약 2분 0초 시점 프레임; 0과 1번 프레임은 제외)
FIRST_ROW_FRAME = 2


@dataclass(frozen=True, eq=False)
class FrameRows:
    """단일 매치의 프레임 시계 질문 행 테이블:
    - k: 정규 타임라인 프레임 번호 (k >= 2)
    - t0: 해당 프레임의 실제 시각 (ms: 예측 모델의 관측 기준점)
    - hi: 발생 관측 윈도우의 종료 시각 (min(t0 + 180초, 경기 종료 - 30초))
    - exposure_ms: 18개 10초 빈 각각의 유효 관측 시간 (ms: 0 ~ 10,000ms)
    """
    k: np.ndarray              # 프레임 번호 (int16)
    t0: np.ndarray             # 관측 시각 (밀리초, ms, int64)
    hi: np.ndarray             # 관측 창 상한 (밀리초, ms, int64)
    exposure_ms: np.ndarray    # (n, 18) 빈별 유효 노출 시간 (밀리초, ms, int32)


@dataclass(frozen=True, eq=False)
class Onsets:
    """특정 프레임 질문 시점(row)과 해당 시점으로부터 3분 이내에 발생한 교전(unit)의 매칭 쌍 테이블:
    - row: 소속 FrameRows 행 인덱스
    - bin: 첫 킬이 발생한 10초 시간 빈 인덱스 (0 ~ 17)
    - mark: 교전의 첫 킬 매치업 표지 (0 ~ 51)
    - unit: 소속 UnitTable 교전 단위 인덱스
    """
    row: np.ndarray            # 질문 행 인덱스 (int32)
    bin: np.ndarray            # 발생 시간 빈 (0~17, int8)
    mark: np.ndarray           # 첫 킬 표지 (0~51, int16)
    unit: np.ndarray           # 교전 단위 인덱스 (int32)


def frame_rows(rec: MatchRecord) -> FrameRows:
    """매치 레코드에서 유효 프레임 시계 행들을 추출하고, 18개 빈에 대한 노출 시간(exposure_ms)을 계산합니다.
    - 2:00 시점(k=2)부터 경기 종료 30초 전까지의 프레임을 추출합니다.
    - 경기 종료 시점에 걸리는 빈은 노출 시간이 10초 미만으로 절단됩니다.
    """
    ts = np.asarray(rec.frame_ts, dtype=np.int64)
    end = rec.T_ms - C.END_CUT_MS

    # 2:00 이후 프레임 중 경기 종료 컷 이전의 프레임 선별
    k = np.arange(FIRST_ROW_FRAME, len(ts), dtype=np.int64)
    k = k[ts[FIRST_ROW_FRAME:] < end]
    t0 = ts[k]

    # 각 프레임의 발생 예측 윈도우 상한(hi) 설정 (최대 180초 또는 경기 종료 30초 전)
    hi = np.minimum(t0 + C.H_OCC_MS, end)

    # 18개 빈(각 10초)의 시작 시각을 계산하고, 유효 관측 구간과 겹치는 노출 시간(0 ~ 10,000ms) 계산
    bin_start = t0[:, None] + C.BIN_MS * np.arange(C.N_BINS, dtype=np.int64)
    exposure = np.clip(hi[:, None] - bin_start, 0, C.BIN_MS)

    return FrameRows(k=k.astype(np.int16), t0=t0, hi=hi, exposure_ms=exposure.astype(np.int32))


def onsets(rows: FrameRows, u: UnitTable) -> Onsets:
    """각 프레임 행(row)의 관측 창 (t0, hi] 내에서 발생한 교전(unit)들을 매칭하여 발생(Onsets) 목록을 구축합니다.
    - 조건: t0 < f <= hi (이미 진행 중인 교전 f <= t0 은 제외하고 새로 시작하는 교전만 집계)
    - 발생 빈 q = (f - t0 - 1) // 10초
    """
    f = u.f
    if f.size > 1 and np.any(f[1:] < f[:-1]):
        raise ValueError('first-kill times must not decrease: the units are in time order')

    # 이진 탐색을 통해 각 행의 창 (t0, hi] 내에 첫 킬 시각 f가 들어오는 교전 인덱스 범위 탐색
    first = np.searchsorted(f, rows.t0, side='right')
    stop = np.searchsorted(f, rows.hi, side='right')
    count = stop - first

    # 행 인덱스와 단위 인덱스를 평탄화(Flatten)하여 1:1 쌍으로 확장
    row = np.repeat(np.arange(len(rows.t0), dtype=np.int64), count)
    start = np.cumsum(count) - count
    unit = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(start - first, count)

    # 첫 킬이 발생한 10초 시간 빈 인덱스 계산
    bin_ = (f[unit] - rows.t0[row] - 1) // C.BIN_MS

    return Onsets(
        row=row.astype(np.int32),
        bin=bin_.astype(np.int8),
        mark=u.mark[unit].astype(np.int16),
        unit=unit.astype(np.int32)
    )
