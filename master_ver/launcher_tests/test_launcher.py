"""공용 합성 실행 계약을 이 프로젝트 설정으로 검증한다."""
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED = PROJECT_ROOT.parent / "tools/project_runtime"
# 테스트 전용 탐색 경로이며 launcher/자식의 PYTHONPATH 환경은 변경하지 않는다.
sys.path.insert(0, str(SHARED / "src"))
sys.path.insert(0, str(SHARED / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from launcher_contract import LauncherContract
from lol_teamfight_master import cli


class LauncherTests(LauncherContract, unittest.TestCase):
    project = cli.PROJECT
    cli = cli
    project_root = PROJECT_ROOT
    distribution = "lol-teamfight-master"
    namespace = "lol_teamfight_master"


if __name__ == "__main__":
    unittest.main()
