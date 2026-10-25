r"""[모듈 개요 및 하네스 위치]
이 모듈은 교전 과정 모형(EPT)과 기준 모형(B0, B1)이 예측하는 조건부 확률법칙(머리, heads)을
합성기(composer, stage S3 Task 9)가 동일한 인터페이스로 호출할 수 있도록 규정하는 핵심 입출력 프로토콜이다 (판정 R3).

A. 하네스 및 파이프라인 내 위치:
   - 교전 발생(M2), 교전 결과(M3), 교전 후 전환(M4) 사슬의 각 단계별 조건부 확률법칙을 추상화한다.
   - B0 GLM (baselines_b0, Task 6), B1 LightGBM (baselines_b1, Task 7), 그리고 본 EPT 신경망 모형 모두
     이 ComposerHeads 프로토콜을 구현하여 합성기에 전달된다.
   - 합성기는 뒤에 어떤 모형이 있는지 인식할 필요 없이 오직 이 규약에 정의된 메서드만을 호출하여
     모의 시뮬레이션을 수행하거나 기댓값을 계산한다.

B. 데이터 바인딩 및 행(Row) 인덱싱 규약:
   - bind(s2, views): 합성기가 평가 대상 시점인 t₀ 분 프레임(frame-clock) 뷰의 행 특징을 계산하여 내부 보관한다.
   - 이후 모든 메서드는 bound 행들의 정수 인덱스 배열 rows (n,)와 발생 사슬 범주들을 입력받아 확률/평균을 반환한다.
   - row_view(rows): M4 전환 머리 입력 행렬을 구성하기 위해 bound 특징을 반환한다.
   - 가린 머리(masked heads) 제약: 합성기가 넘겨주는 사슬의 선행 정보만을 조건으로 취하며, 미래 사건이나
     누설 가능성이 있는 정보는 배제된다 (판정 P6).

C. 출력 텐서 규격 (OUTPUT_WIDTHS):
   - rate(rows)                         : (n, 18)    초당 첫 킬 발생률 (구간 0~17)
   - mark_prob(rows, q)                 : (n, 52)    첫 킬 표지(선제킬 쌍 50개 + 처형 2개) 확률분포 (합=1)
   - zone_prob(rows, q, m)              : (n, 10)    첫 킬 구역 확률분포 (합=1, 구역 0~9)
   - size_prob(rows, q, m, z)           : (n, 4)     교전 규모 범주 확률분포 (합=1, 규모 0~3)
   - kill_rate(rows, q, m, z, s)        : (n, 2)     팀별(Blue, Red) 평균 킬 수 (Poisson 평균)
   - gold_prob(rows, q, m, z, s, layer) : (n, 2, 31) 킬 수 층(layer 1~6) 조건부 팀별 킬 골드 구간(0~30) 확률분포
   - obj_rate(rows, q, m, z, s, K)      : (n, 18)    18개 오브젝트 칸별 기대 획득 수 (Poisson 평균)
   - conv_rate(m4_inputs)               : (n, 18)    18개 오브젝트 칸별 초당 전환율 (Poisson 발생률)

D. 핵심 수식 (HEAD-1 ~ HEAD-5):
   (HEAD-1)  n_q ~ Poisson(λ_q · Δ_q)                                 구간 q의 발생 수 (λ_q = rate[:, q], Δ_q = 노출 시간(초))
   (HEAD-2)  P(m, z, s | q) = π(m | q) · ρ(z | q, m) · P(s | q, m, z)   표지·구역·규모 조건부 사슬
   (HEAD-3)  k_blue ~ Poisson(μ_blue),  k_red ~ Poisson(μ_red)        팀별 킬 수 포아송 기댓값
             킬 k ≥ 1인 팀의 골드 구간 ~ gold_prob(layer = min(k, 6)); k = 0이면 골드 0 확정
   (HEAD-4)  O_c ~ Poisson(μ_c);  c칸 전환 수 ~ Poisson(λ_c · e)      오브젝트 및 전환 포아송 분포
   (HEAD-5)  p = (1 − ν) · p_model + ν / C,   ν = 1e-6                 범주 출력 균등 분포 섞기 (판정 R13, NLL 무한대 방지)

E. 하네스 제약 및 공통 가정:
   - 모든 범주형 출력은 전폭(full-width) 확률 벡터이며 strictly positive이다 (HEAD-5 적용).
   - analytic_ok 플래그: B0와 같이 모형이 닫힌 꼴(closed-form) 가치 기댓값을 제공할 수 있는지 표시 (B0: True, B1/EPT: False).
   - obj_uses_K 및 conv_uses_KO: M3 오브젝트 및 M4 전환 머리가 K와 O를 조건부로 받는지 여부 명시.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P4 (the factorisation P(s) P(K | s) P(O | K, s) and the probability family),
#      P6 (what the masked heads are conditioned on) and Task 9 (the composer); controller ruling R3 (this module) and R13 (full-width,
#      strictly positive categorical outputs).
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from c_ept import config as C

# =============================================================================
# 블록 1: 전역 상수 및 출력 형상 규격 (OUTPUT_WIDTHS)
# =============================================================================

# 지원하는 특징 규격 ('b0': 39개 열, 'b1': 요약 통계량 포함 열)
FEATURE_KINDS = ('b0', 'b1')

# 각 머리별 반환 배열의 마지막 차원 크기 규약.
# 단위 테스트에 의해 units, nll, features, targets 모듈의 상수와 일치함이 검증됨.
OUTPUT_WIDTHS = {
    'rate': (C.N_BINS,),                 # (18,) 10초 간격 18개 발생 구간
    'mark_prob': (C.N_MARKS,),           # (52,) 선제 킬 쌍 50개 + 처형 2개
    'zone_prob': (10,),                  # (10,) 구역 0 .. 9 (관측 구역)
    'size_prob': (4,),                   # (4,) 소/중/대/초대형 교전 규모 0 .. 3
    'kill_rate': (2,),                   # (2,) Blue, Red 팀별 평균 킬 수
    'gold_prob': (2, 31),                # (2, 31) Blue, Red 팀별 31개 킬 골드 구간
    'obj_rate': (2 * len(C.O_KINDS),),   # (18,) 9종 오브젝트 x 2개 팀 = 18개 칸
    'conv_rate': (2 * len(C.O_KINDS),),  # (18,) 18개 오브젝트 칸 초당 전환율
}


# =============================================================================
# 블록 2: 합성기 연동 ComposerHeads 프로토콜 정의
# =============================================================================

@runtime_checkable
class ComposerHeads(Protocol):
    """합성기(Composer)가 교전 발생 및 결과를 순차 모의/평가하기 위해 읽는 통합 머리 인터페이스.

    [프로토콜 역할]:
    - isinstance(x, ComposerHeads)를 통해 런타임에 모형 구현체의 규약 충족 여부를 검증한다.
    - B0 GLM, B1 LightGBM, EPT 신경망 모형이 모두 이 프로토콜을 구현하여 동일 규격으로 연동된다.
    - 식 (HEAD-1) ~ (HEAD-4)의 사슬을 단계별 조건부 호출로 분해하여 계산한다.
    """
    feature_kind: str      # 사용 중인 행 특징 종류 ('b0' 또는 'b1')
    analytic_ok: bool       # 닫힌 꼴(closed-form) 교전 가치 계산 가능 여부 (B0: True, B1/EPT: False)
    obj_uses_K: bool        # M3 오브젝트 머리가 킬 결과 K를 조건으로 받는지 여부
    conv_uses_KO: bool      # M4 전환 머리가 K와 O를 조건으로 받는지 여부

    def bind(self, s2, views) -> None:
        """평가 대상인 t₀ 분 프레임(frame-clock) 뷰들을 내부 행 특징으로 계산하여 바인딩한다.

        [역할]:
        - 합성기가 한 번 바인딩해 두고 이후의 다단계 사슬(rate -> mark -> zone -> size ...)에서
          행 인덱스 `rows`만으로 특징을 재사용하도록 캐싱한다.
        - 가린 머리(masked heads)의 정보 누설을 방지하기 위해 사건 뷰가 아닌 분 프레임 뷰만 읽는다.

        [입력]:
        - s2: S2 저장소 객체 (c_ept.store_core.Store)
        - views: 바인딩할 S2 프레임 뷰 번호 리스트 또는 bool 마스크
        """
        ...

    def row_view(self, rows) -> np.ndarray:
        """바인딩된 t₀ 행들의 원시 특징 배열을 반환한다.

        [역할]:
        - 합성기가 M4 전환 머리의 입력 행렬(features.m4_masked_inputs)을 구성할 때 호출한다.

        [입력]:
        - rows: 바인딩된 뷰 내 행 위치를 가리키는 정수 인덱스 배열 (n,)

        [출력]:
        - (n, d) float32 행 특징 행렬 (b0: d=39, b1: d=440 등)
        """
        ...

    def rate(self, rows) -> np.ndarray:
        """구간 q (0 .. 17)별 첫 킬의 초당 발생률 λ_q를 반환한다.

        [역할]:
        - 식 (HEAD-1) n_q ~ Poisson(λ_q · Δ_q)의 발생률 모수를 18개 구간 전체에 대해 계산.

        [입력]:
        - rows: (n,) 바인딩된 프레임 행 인덱스

        [출력]:
        - (n, 18) float64: 각 구간의 초당 첫 킬 발생률 (≥ 0)
        """
        ...

    def mark_prob(self, rows, q) -> np.ndarray:
        """주어진 구간 q에서 발생한 첫 킬의 표지 m (0 .. 51)에 대한 조건부 확률분포를 반환한다.

        [역할]:
        - 식 (HEAD-2)의 π(m | q) 계산. 선제 킬 50방향 쌍 + 처형 2개 = 총 52개 범주.

        [입력]:
        - rows: (n,) 바인딩된 프레임 행 인덱스
        - q: (n,) 발생 구간 인덱스 (0 .. 17)

        [출력]:
        - (n, 52) float64: 합이 1인 확률분포 (HEAD-5 균등 섞기 적용, strictly positive)
        """
        ...

    def zone_prob(self, rows, q, m) -> np.ndarray:
        """주어진 구간 q와 표지 m에 조건부인 첫 킬 구역 z (0 .. 9)의 확률분포를 반환한다.

        [역할]:
        - 식 (HEAD-2)의 ρ(z | q, m) 계산. 관측 구역 10개에 대한 조건부 확률.
        - 위치 미관측(z = -1)은 이 분포의 결과가 아니며 관측된 10개 구역에 대해서만 합이 1로 정규화됨.

        [입력]:
        - rows: (n,) 바인딩된 프레임 행 인덱스
        - q: (n,) 발생 구간 인덱스 (0 .. 17)
        - m: (n,) 첫 킬 표지 인덱스 (0 .. 51)

        [출력]:
        - (n, 10) float64: 합이 1인 구역 확률분포 (HEAD-5 적용)
        """
        ...

    def size_prob(self, rows, q, m, z) -> np.ndarray:
        """주어진 q, m, z에 조건부인 교전 규모 s (0 .. 3)의 확률분포를 반환한다.

        [역할]:
        - 식 (HEAD-2)의 P(s | q, m, z) 계산.
        - 구역 미관측(z = -1)은 전용 수준(level 0)으로 매핑되어 안전하게 처리됨.

        [입력]:
        - rows: (n,) 바인딩된 프레임 행 인덱스
        - q: (n,) 발생 구간 인덱스 (0 .. 17)
        - m: (n,) 첫 킬 표지 인덱스 (0 .. 51)
        - z: (n,) 첫 킬 구역 인덱스 (-1 .. 9)

        [출력]:
        - (n, 4) float64: 규모 s (0: 1킬, 1: 2킬, 2: 3~4킬, 3: 5킬 이상) 확률분포 (합=1)
        """
        ...

    def kill_rate(self, rows, q, m, z, s) -> np.ndarray:
        """주어진 q, m, z, s에 조건부인 Blue 및 Red 팀의 기대 킬 수 (μ_blue, μ_red)를 반환한다.

        [역할]:
        - 식 (HEAD-3) k_blue ~ Poisson(μ_blue), k_red ~ Poisson(μ_red)의 포아송 기댓값 계산.

        [입력]:
        - rows: (n,) 바인딩된 프레임 행 인덱스
        - q: (n,) 발생 구간 (0 .. 17)
        - m: (n,) 첫 킬 표지 (0 .. 51)
        - z: (n,) 첫 킬 구역 (-1 .. 9)
        - s: (n,) 교전 규모 (0 .. 3)

        [출력]:
        - (n, 2) float64: [Blue 기대 킬 수, Red 기대 킬 수] (각 값 ≥ 0)
        """
        ...

    def gold_prob(self, rows, q, m, z, s, layer) -> np.ndarray:
        """킬 수 층(layer)에 조건부인 각 팀의 킬 골드 구간(0 .. 30) 확률분포를 반환한다.

        [역할]:
        - 식 (HEAD-3)의 gold_prob 계산. 킬이 k ≥ 1인 팀에 대해 31개 골드 구간의 확률을 예측.
        - layer: 팀별 킬 수 층 (1 .. 6, 6 = 6킬 이상). k = 0인 팀은 골드 0 확정이므로 호출 제외됨.

        [입력]:
        - rows: (n,) 바인딩된 프레임 행 인덱스
        - q, m, z, s: (n,) 발생 사슬 조건
        - layer: (n, 2) 각 팀의 킬 수 층 (1 .. 6)

        [출력]:
        - (n, 2, 31) float64: [Blue 31구간 확률, Red 31구간 확률] (각 행 합=1, HEAD-5 적용)
        """
        ...

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        """교전 중 획득되는 18개 오브젝트 칸별 기대 획득 수 μ_c를 반환한다.

        [역할]:
        - 식 (HEAD-4) O_c ~ Poisson(μ_c)의 포아송 기댓값 계산 (노출 시간 없는 개수 자체의 기댓값).
        - obj_uses_K가 False인 모형(B0)은 K를 무시하고, True인 모형(B1, EPT)은 K를 조건으로 반영함.

        [입력]:
        - rows: (n,) 바인딩된 프레임 행 인덱스
        - q, m, z, s: (n,) 발생 사슬 조건
        - K: (n, 4) float64: [Blue 킬 수, Red 킬 수, Blue 킬 골드(원수치), Red 킬 골드(원수치)]

        [출력]:
        - (n, 18) float64: 18개 오브젝트 칸별 기대 획득 수 (≥ 0)
        """
        ...

    def conv_rate(self, m4_inputs) -> np.ndarray:
        """교전 종료 후 전환 창(60초) 동안 획득되는 18개 오브젝트 칸별 초당 전환율 λ_c를 반환한다.

        [역할]:
        - 식 (HEAD-4) 전환 수 ~ Poisson(λ_c · e)의 초당 발생률 모수 계산.
        - features.m4_masked_inputs로 구성된 가린 M4 입력 행렬을 직접 전달받아 추론 수행.

        [입력]:
        - m4_inputs: (n, d_m4) float32 행렬 (t₀ 특징, q, m, z, K, O 결합)

        [출력]:
        - (n, 18) float64: 18개 오브젝트 칸별 초당 전환율 (≥ 0)
        """
        ...
