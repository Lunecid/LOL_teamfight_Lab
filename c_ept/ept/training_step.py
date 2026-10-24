"""단일 배치의 순전파, 손실 평가, 역전파 및 최적화 단계 실행 모듈 (LOSS-4, TRN-7).

책임 한계:
- 배치 로딩, GPU 전송, 학습률 스케줄링, 그래디언트 초기화(zero_grad), 체크포인트 주기는 train 루프 소유다.
- 주입된 scorer 및 클록 인터페이스를 통해 계산 순서와 구간별 시간 측정 경계를 보존한다. optimizer와 components는 변경된다."""
from __future__ import annotations

import torch
from c_ept.ept import loss


def optimize_batch(fa, batch, optimizer, scales, step: int, components: dict, *, score, clock):
    """배치 순전파, 손실 계산, 역전파, 옵티마이저 갱신을 단일 단계로 실행한다 (TRN-7).

    매개변수:
    - fa: 특성 추출 및 인코딩 모델 인터페이스.
    - batch: 현재 훈련 배치 데이터.
    - optimizer: 파이토치 옵티마이저.
    - scales: total_loss에 전달할 손실 정규화 척도.
    - step: 전역 훈련 스텝 인덱스.
    - components: 연산 구간별 소요 시간 누적 딕셔너리 (forward_seconds, backward_seconds, optimizer_seconds).
    - score: 점수 계산 콜백 (키워드 전용).
    - clock: 동기화 시간 측정 콜백 (키워드 전용).

    반환값:
    - 계산된 total 손실 텐서.

    예외:
    - 식 (LOSS-4)를 통해 집계된 total 손실이 유한하지 않은 경우(NaN/Inf) FloatingPointError를 발생시킨다.
    - score(fa, fa.encode(batch), batch)의 각 항에서 손실과 타깃 사전을 읽어 total_loss를 호출한다.
    - 비유한 손실이면 backward/step을 수행하지 않는다. 반환 텐서는 detach하지 않는다."""
    component_start = clock()
    scored = score(fa, fa.encode(batch), batch)
    total = loss.total_loss({k: v[2] for k, v in scored.items()}, scales, {k: v[1] for k, v in scored.items()})
    if not bool(torch.isfinite(total)):
        raise FloatingPointError(f'the training loss is not finite at step {step}')
    components['forward_seconds'] += clock() - component_start
    component_start = clock()
    total.backward()
    components['backward_seconds'] += clock() - component_start
    component_start = clock()
    optimizer.step()
    components['optimizer_seconds'] += clock() - component_start
    return total
