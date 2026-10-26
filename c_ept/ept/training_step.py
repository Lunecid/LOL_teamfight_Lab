"""단일 배치의 순전파, 손실 평가, 역전파 및 최적화 단계 실행 모듈 (LOSS-4, TRN-7).

책임 한계:
- 배치 로딩, GPU 전송, 학습률 스케줄링, 그래디언트 초기화(zero_grad), 체크포인트 주기는 train 루프 소유다.
- 주입된 scorer 및 클록 인터페이스를 통해 계산 순서와 구간별 시간 측정 경계를 보존한다. optimizer와 components는 변경된다."""
from __future__ import annotations

import torch
from c_ept.ept import loss


def terrain_penalty(fa):
    """The smoothness penalty R_Ω of the model ``fa`` (a 0-d tensor, ENC-7): the sum of ``TerrainField.penalty()`` over the terrain fields in ``fa.modules()`` (the EPT and N2 each own one),
    or a plain 0.0 for a model without one (no field module). A field without learned channels (``field_channels=0``) gives 0, with no gradient.

    해석: 지형장은 모형 안에 하나뿐이므로 합은 곧 그 장의 벌점이다. 장이 없으면 벌점이 없고 오류도 아니다(가중이 0이 아닌데 학습 채널이 하나도 없는 설정은 ``train``이 일을 시작하기 전에
    거부한다). ``encoder``는 가중이 0이 아닐 때만 가져온다: 기본 경로는 이 모듈의 가져오기도 부르는 쪽의 모형도 건드리지 않는다."""
    from c_ept.ept import encoder
    fields = [m.penalty() for m in fa.modules() if isinstance(m, encoder.TerrainField)]
    return sum(fields[1:], fields[0]) if fields else 0.0


def optimize_batch(fa, batch, optimizer, scales, step: int, components: dict, *, score, clock, guards=None, smooth_weight: float = 0.0):
    """배치 순전파, 손실 계산, 역전파, 옵티마이저 갱신을 단일 단계로 실행한다 (TRN-7).

    수식
    ----
    (LOSS-5)  L = Loss_EPT + λ_Ω · R_Ω      (λ_Ω = smooth_weight; λ_Ω = 0이면 L = Loss_EPT이고 R_Ω는 계산하지 않는다)

      기호      | 뜻                                                    | 단위     | 범위     | 코드 변수
      Loss_EPT  | 배치의 총손실 (LOSS-4)                                | —        | ≥ 0      | total
      R_Ω       | 학습 지형장의 매끄러움 벌점 (ENC-7, 가중 없음)        | 표의 값² | ≥ 0      | penalty, components['terrain_penalty']
      λ_Ω       | 지형 매끄러움 가중 (저자 결정 D8)                     | —        | ≥ 0      | smooth_weight

    해석: 역전파하고 유한성을 검사하는 손실은 L이지만 돌려주는 값은 데이터 손실 Loss_EPT이다(적합 곡선이 λ_Ω와 상관없이 같은 양을 가리키도록). 벌점은 매개변수만의 함수여서 배치와 무관하고,
    걸음마다 순전파 구간(forward_seconds) 안에서 한 번 계산해 그 값을 ``components['terrain_penalty']``에 더한다(가중이 0일 때는 이 키를 만들지 않아 기본 경로의 components는 이전과 같다).
    L이 유한하지 않으면(벌점이 넘친 경우도) 역전파도 옵티마이저 걸음도 하지 않는다. STOP 평가는 이 함수를 거치지 않으므로 벌점이 들어가지 않는다.

    매개변수:
    - fa: 특성 추출 및 인코딩 모델 인터페이스.
    - batch: 현재 훈련 배치 데이터.
    - optimizer: 파이토치 옵티마이저.
    - scales: total_loss에 전달할 손실 정규화 척도.
    - step: 전역 훈련 스텝 인덱스.
    - components: 연산 구간별 소요 시간 누적 딕셔너리 (forward_seconds, backward_seconds, optimizer_seconds). ``smooth_weight``가 0이 아니면 이 걸음의 R_Ω가 'terrain_penalty' 키에 더해진다.
    - score: 점수 계산 콜백 (키워드 전용).
    - clock: 동기화 시간 측정 콜백 (키워드 전용).
    - guards: ``step_guard.StepGuards`` of a trusted step (S4b speed-ups item 3, D32; keyword only, default None = the checks and the finiteness read of every earlier run). The caller runs this
      function inside ``step_guard.trust(guards)`` with a batch made by ``batch_prep.Preparation``; the model checks that the loss functions deferred into ``guards`` and the finiteness of L are
      read in ONE transfer before backward (``guards.check``): the same refusals (type, message, count, order) as before, no backward and no optimizer step after one.
      해석: 걸음마다 읽기를 모형 점검 수만큼에서 한 번으로 줄이되 막히는 조건·오류·시점(역전파 앞)은 그대로이다. 걸음에 준 ``guards``가 지금 열린 ``step_guard.trust`` 범위의 것이 아니면(범위 없이 guards를 주거나,
      범위 안에서 guards를 안 주거나 다른 것을 줌) RuntimeError로 닫는다: 미룬 점검을 아무도 읽지 않는 열린 상태를 막는다 (검토 m1).
      다중 결함의 순서(검토 m2, 수용한 변경): 자료 결함은 이제 인코더와 모형 점검보다 먼저(``Preparation``, 커널에서는 생산 스레드의 ``next()``) 거부된다. 자료와 모형이 함께 틀린 배치에서는 자료 오류가 이기고, 보기 텐서와
      행 필드가 함께 틀리면 행 필드 오류가 인코더 오류보다 먼저 나올 수 있다. 거부는 ``zero_grad``·학습률 지정·``transfer_seconds`` 집계 앞에서 일어난다. 같은 걸음에서 닫히고 가중은 바뀌지 않는다(결함 실행이 같은 걸음 수·손실·매개변수를 보인다).
    - smooth_weight: λ_Ω (키워드 전용, 기본값 0.0 = 이전 실행과 비트까지 같은 경로).

    반환값:
    - 계산된 total 손실 텐서 (벌점을 더하기 전의 데이터 손실).

    예외:
    - 식 (LOSS-4)를 통해 집계된 total 손실이 유한하지 않은 경우(NaN/Inf) FloatingPointError를 발생시킨다.
    - score(fa, fa.encode(batch), batch)의 각 항에서 손실과 타깃 사전을 읽어 total_loss를 호출한다.
    - 비유한 손실이면 backward/step을 수행하지 않는다. 반환 텐서는 detach하지 않는다."""
    from c_ept.ept import step_guard         # at the call, not at the import: the tests load this file with a stand-in torch
    if step_guard.current() is not guards:
        # Fail closed (review m1): inside trust(g) the loss functions defer their model checks into g and only g.check reads them; a step run in another scope than the guards it is given would skip them.
        raise RuntimeError('optimize_batch must get the StepGuards of the active step_guard.trust scope (guards=...), and none outside a trust scope')
    component_start = clock()
    scored = score(fa, fa.encode(batch), batch)
    total = loss.total_loss({k: v[2] for k, v in scored.items()}, scales, {k: v[1] for k, v in scored.items()})
    objective = total
    if smooth_weight != 0.0:
        penalty = terrain_penalty(fa)
        objective = total + smooth_weight * penalty                # (LOSS-5): the loss that is differentiated
        # the record is a plain number: detached first (the loss term above keeps its graph); a model without a field gives the float 0.0, which has nothing to detach
        components['terrain_penalty'] = components.get('terrain_penalty', 0.0) + float(penalty.detach() if torch.is_tensor(penalty) else penalty)
    if guards is None:
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError(f'the training loss is not finite at step {step}')
    else:
        guards.check(torch.isfinite(objective), step)      # (GRD-1)..(GRD-3): the model checks of the step and the finiteness of L, ONE read, before backward
    components['forward_seconds'] += clock() - component_start
    component_start = clock()
    objective.backward()
    components['backward_seconds'] += clock() - component_start
    component_start = clock()
    optimizer.step()
    components['optimizer_seconds'] += clock() - component_start
    return total
