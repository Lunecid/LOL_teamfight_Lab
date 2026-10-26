"""Profile-driven run kernel (S4b preparation, track K, decision D11).

One kernel runs the training, evaluation and measurement runs of S4b from hash-locked declarative profiles. The old stacks (S3, S4a,
the T003 operator, the T008/T009 comparison) stay as they are and are frozen. This package starts with ``run.io``, the one strict
JSON and hash input/output that every later module of the kernel (identity, profile, guards, ledger, supervisor, runners) reads
and writes through. The package itself imports nothing, so that ``run.io`` can be used before any model code is loaded.

해석: 실행 핵심은 설정을 코드가 아니라 해시로 잠근 프로파일 파일로 받는다. 이 패키지의 첫 부품은 모든 부품이 같이 쓰는 엄격한 JSON·해시
입출력(``run.io``)이며, 패키지 자체는 아무것도 import하지 않는다.
"""
