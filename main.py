#!/usr/bin/env python3
"""LOL Teamfight Prediction — Entry Point

Usage:
    python main.py                          # 기본 실행 (mode=all)
    python main.py --mode build_cache       # 캐시만 빌드
    python main.py --mode train             # 학습만
    python main.py --mode all --seed 42     # 시드 지정
    python main.py --models rnn,gnn         # 모델 지정

이 파일은 runner.py의 main()을 직접 호출합니다.
lol_teamfight 래퍼 패키지 없이 루트 레벨에서 독립 실행됩니다.
"""

from runner import main

if __name__ == "__main__":
    main()
