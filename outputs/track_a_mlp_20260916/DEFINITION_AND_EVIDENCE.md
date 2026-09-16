# 정의·근거·상태 (Track A MLP, 2026-09-16)

## 진행 전/후

| 요소 | 이전 | 이번 | 근거 유형 |
|---|---|---|---|
| 교전 모집단·라벨·분할 | iq와 동일(동결) | 변경 없음 | 부모 계약(fc/cr) |
| 입력 | iq 352열 | 동일 352열 | 부모 스키마 hash |
| 학습기 | PT·logistic·LightGBM | + plain MLP, residual MLP | DELTA_Q 프로토콜 §3, COG_MODEL_COVERAGE §2 (우리 운영 선택) |
| 정지·재적합 | iq LightGBM stop10 규칙 | 같은 stop10 배정, epoch 단위 patience 10, 전체 TRAIN 재적합 | 프로토콜 §3-2·3 |
| 예측 정의 | sklearn/LightGBM CPU | 저장 float32 가중치의 CPU float64 추론 | 재현성 설계 선택 |
| 비교·부트스트랩 | 5 대비 | 6 대비(주: residual MLP − LightGBM) | 프로토콜 §5 |

## 주요 수치 근거

- MAIN TEST T all PRIMARY: residual MLP - full LightGBM: ΔBrier +0.00061 [+0.00012, +0.00110] — `eval/results.json` → results.MAIN_TEST.T.bootstrap.all.pairs[0]
- MAIN TEST T B40 PRIMARY: residual MLP - full LightGBM: ΔBrier +0.00010 [-0.00135, +0.00132] — `eval/results.json` → results.MAIN_TEST.T.bootstrap.B40.pairs[0]
- MAIN TEST N all PRIMARY: residual MLP - full LightGBM: ΔBrier +0.00110 [+0.00079, +0.00145] — `eval/results.json` → results.MAIN_TEST.N.bootstrap.all.pairs[0]
- MAIN TEST N B40 PRIMARY: residual MLP - full LightGBM: ΔBrier +0.00185 [+0.00123, +0.00241] — `eval/results.json` → results.MAIN_TEST.N.bootstrap.B40.pairs[0]

## 무결성

- frozen_manifest sha256 6a2fafe6fde0fb97433717ec6cb05741f38efd00bfada781dc9781479164833d; results sha256 d4d4e4f3d7324f4b28fff594dc95e20527cca62afaf1072c658a0e3f9fdc2d45; validation 115검사 실패 0.
- 부모 read-only, 새 root만 기록, 봉인 세트는 이 실행의 동결 이후 접근, 학습·동결·평가 소스 hash 동결 manifest에 기록.
