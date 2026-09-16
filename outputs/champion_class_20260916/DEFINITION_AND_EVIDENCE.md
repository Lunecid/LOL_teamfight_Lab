# 정의·근거·상태 (챔피언 클래스 특징, 2026-09-16)

| 요소 | 이전 | 이번 | 근거 |
|---|---|---|---|
| 챔피언 정보 in q | 없음(p_pre 경유 가산항뿐) | 슬롯 태그·클래스 집계·클래스 쌍·정체성 one-hot 팔 | 명세 §Feature blocks |
| 클래스 출처 | — | Data Dragon tags 6종, 패치별 표(sha 고정) | ddragon/fetch_manifest.json |
| 구성 | iq h90 승자 | 고정(재선정 없음), 팔마다 보정만 선택 | 프로토콜 규칙 |
| 주 비교 | — | T · LightGBM · class_pairs − base · 전체 · Brier | protocol.json evaluation.primary |

- frozen_manifest sha256 1909f76aa172d17ffeb7fd374e44359e833de4d1843d18b22778efdfab8da7d5; results sha256 9b4568ef2dc526f6a815e05d029cd736a74b4ee581e5311cd5c50895373d839d; validation 91검사 실패 0.
