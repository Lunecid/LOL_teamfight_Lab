"""C 모델 S4a의 EPT(Engagement Process Transformer) 부품.

data는 배치, encoder는 관계 부호화, heads는 머리와 프레임 평균, loss는 손실,
scales는 B0 정규화 척도를 맡는다. 구현된 학습 고리는 train이며 checkpoints는
체크포인트 IO, training_step은 배치 계산, c_ept.runtime.resources는 자원 측정을 맡는다.
모듈은 from c_ept.ept import data와 같이 연결하고 S1/S2 코드는 c_ept에서 읽는다.

역사적 설계: docs/superpowers/specs/2026-10-20-c-ept-design.md 3–7절.
역사적 계획: docs/superpowers/plans/2026-10-22-c-ept-s4a.md.
두 문서는 이 checkout에 미동봉되어 있어 이번 검토에서 원문을 검증하지 않았다.
현재 수식 등록부 경로는 docs/superpowers/notes/equation-index.md이며 존재 확인과 정확성 검증은 구별한다."""
