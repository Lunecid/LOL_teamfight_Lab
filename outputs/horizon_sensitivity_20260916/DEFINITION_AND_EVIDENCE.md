# 정의·근거·상태 (종료 상한 민감도, 2026-09-16)

| 요소 | 이전 | 이번 | 근거 |
|---|---|---|---|
| 상한 | h90 주 분석(iq, Track A) | h60·h120 재적합 | DELTA_Q 프로토콜 §3-6 |
| 구성 | h90에서 선정 | 고정(재선정 없음), 보정만 선택 | 프로토콜 규칙 |
| 라벨 | Y_h90 | Y_h60, Y_h120(부모 라벨, 동일 종료 규칙) | 부모 계약 |
| 행 | h90 유효 | 상한별 유효 = 동일 수(코호트 manifest) | cohort_manifest.json |

- frozen_manifest sha256 d99561e1ad6a501f1a8a4df6d65487b0b5a3069b362e3504a8943daf3723f6b4; results sha256 3daa885346c420fcf4bf9bee8d768e251fdc81247b484ae41ff293bcd39e0e07; validation 91검사 실패 0.
