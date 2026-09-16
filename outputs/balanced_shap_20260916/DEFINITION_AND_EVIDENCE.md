# 정의·근거·상태 (균형 상태 SHAP, 2026-09-16)

| 요소 | 이전 | 이번 | 근거 |
|---|---|---|---|
| 설명 대상 | 역할 모델(cr), 전체 q(fc) 256사례 | 최종 선정 q(iq LightGBM)와 plain MLP, 전체·B40 각 256사례 | DELTA_Q 프로토콜 §6 |
| 그룹 | fc 8그룹(챔피언 포함) | 352열 7그룹(챔피언 제외) | 부모 shap_groups 제한 |
| 배경 | TRAIN 128 | 코호트 TRAIN 128, 셀·모델 공통 | 프로토콜 |

- frozen_manifest sha256 5a192ac0eb4fdc209f1d03bad358a72410912b5ac4431efc73f3189862311aae; shap_summary sha256 1df1e591a9fa3abc0a06a8f7485495b464be41e6965c64e5b13d198e119f4f96; validation 36검사 실패 0.
