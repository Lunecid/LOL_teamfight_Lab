# 정의·근거·상태 (개발 패치 전용 정의 재탐지, 2026-09-16)

| 요소 | 동결 | 이번(dev) | 근거 |
|---|---|---|---|
| G | 13.7 s (pooled 13.7246) | 14.0 s (15.14 13.9637) | spec_pooled / spec_15.14 |
| D | 4,264 u (pooled 4263.87) | 4,285 u (15.14 4285.26) | 같은 반올림 규칙 |
| 탐지 코드 | 부모 | 동일(동결 상수 재현 정확) | redetect/frozen_check.json |
| 추출·라벨 | 부모 | 동일 함수·동결 V (fixture 비트 일치) | rebuild/fixture_check.json |

- frozen_manifest sha256 f30e15e2ab2a533728ed1acb2685ff7400dc376237b11e3448389b7c8d9d782f; results sha256 ad80fcd3f0c08f25f78b7358e0581d20092ad51e780318031392034d7416104c; validation 53검사 실패 0.
