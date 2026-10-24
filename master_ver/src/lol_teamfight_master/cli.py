"""계보별 설정만 유지한다. 검증/명령 구성/실행은 공용 runtime이 담당한다."""
from typing import Optional, Sequence

from lol_project_runtime.cli import main as runtime_main
from lol_project_runtime.launcher import Project, SourceEntry

PROJECT = Project("Master", ("pyproject.toml", "CODEMAP.md",
                              "studies/v_production_20261004/train.py",
                              "studies/v_production_20261004/worker.py"),
                  (SourceEntry("v-worker", "studies/v_production_20261004/worker.py"),
                   SourceEntry("v-pipeline", "reports/v_production_20261004/pipeline.py",
                               reference_only=True)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """기본 호출은 공용 도움말만 표시하고 연구 실행은 명시적 run으로 제한한다."""
    return runtime_main(PROJECT, argv)
