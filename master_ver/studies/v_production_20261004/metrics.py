"""수치적으로 안정화된 float64 이진 분류 손실 및 활성화 연산을 제공하는 독립 평가 모듈.

기존 연산 순서를 유지하면서, 단일 행 기준 비가중 산술 평균 기반의 이진 교차 엔트로피(Log Loss)와 시그모이드 변환을 제공합니다."""
from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """수치적 오버플로를 방지하도록 분기 처리된 float64 기반 1차원/다차원 시그모이드 함수.

    x >= 0 구간은 1 / (1 + exp(-x)), x < 0 구간은 exp(x) / (1 + exp(x))를 적용하여 거대 양수/음수 입력에서도 부동소수점 오버플로 없이 안정적인 확률 배열을 반환합니다.
    입력 형태를 유지합니다. 입력 유한성 검사는 하지 않으며 ±Inf는 각각 1/0, NaN은 NaN으로 남습니다."""
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def raw_logloss_from_logits(logits: Any, target: Any) -> float:
    """로짓과 이진 타깃으로부터 logaddexp를 이용해 계산하는 수치 안정 단위 행 이진 로그 손실.

    수식 mean(logaddexp(0, logits) - target * logits)를 계산합니다. float64로 변환하여 각각 평탄화한 logits와 target 길이가 불일치하거나, 비유한 값(NaN/Inf)가 존재하거나, 타깃에 0.0과 1.0 외 값이 있거나, 빈 입력인 경우 ValueError를 발생시킵니다.
    모든 입력 행에 같은 가중치를 주며 경기 단위 가중 평균이나 경기별 평균은 아닙니다."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.shape != labels.shape or not np.isfinite(values).all() or not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("logits/target must be aligned finite binary data")
    if not len(values):
        raise ValueError("logloss requires at least one row")
    return float(np.mean(np.logaddexp(0.0, values) - labels * values))


def raw_logloss_from_probability(probability: Any, target: Any) -> float:
    """예측 확률값과 이진 타깃 간 단위 행 로그 손실을 클리핑 및 역로짓 변환 후 계산합니다.

    확률값 내 비유한 값(NaN/Inf) 또는 (0, 1) 경계 밖 값이 감지되면 [1e-12, 1 - 1e-12] 구간으로 클리핑한 뒤, log(p) - log1p(-p)로 로짓을 복원하여 raw_logloss_from_logits를 호출합니다.
    이는 기존 동작으로, ±Inf와 범위 밖 유한 값도 경계로 잘립니다. NaN은 남아 후속 로짓 검사에서 거부됩니다. 정상 열린 구간 입력은 별도로 자르지 않습니다."""
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        values = np.clip(values, 1e-12, 1.0 - 1e-12)
    return raw_logloss_from_logits(np.log(values) - np.log1p(-values), target)
