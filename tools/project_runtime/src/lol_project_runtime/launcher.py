"""명시한 연구 소스 루트의 파일 가용성을 검사하고 자식 프로세스 실행 계획을 만든다.

계획 구성 시 연구 코드를 import하지 않으며, 고정 진입점과 해석된 경로의 루트 포함 관계를 확인한다.
이는 보안 샌드박스나 소스 해시 인증이 아니다. execute_plan은 지정한 코드를 현재 사용자 권한으로
실행하므로 신뢰하는 소스 트리를 지정해야 한다. 실행 시점까지 파일 불변성을 보장하지 않는다."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Sequence, Tuple


@dataclass(frozen=True)
class SourceEntry:
    """고정 상대 경로만 허용한다. reference는 배포 사본에서 빠질 수 있다."""
    name: str
    relative_path: str
    reference_only: bool = False


@dataclass(frozen=True)
class Project:
    name: str
    markers: Tuple[str, ...]
    entries: Tuple[SourceEntry, ...]


@dataclass(frozen=True)
class LaunchPlan:
    root: Path
    command: Tuple[str, ...]


def _contained_file(root: Path, relative_path: str) -> Path:
    """심볼릭 링크로 루트 밖 파일을 실행하는 경우도 차단한다."""
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("source path escapes root: " + relative_path)
    if not candidate.is_file():
        raise ValueError("source file unavailable: " + relative_path)
    return candidate


def validate_source_root(project: Project, source_root: str) -> Path:
    """프로젝트 소스 루트 절대 경로 및 필수 표식(marker) 파일 존재 여부를 검증합니다.

    source_root가 명시적 절대 경로인지, 실제 디렉터리인지 확인하고 각 마커 파일이 루트 경계를 탈출하지 않는지 검사합니다. 파일 내용이나 과학적 유효성은 검증하지 않으며 위반 시 ValueError를 발생시킵니다."""
    root = Path(source_root).expanduser()
    if not root.is_absolute():
        raise ValueError("--source-root must be an explicit absolute project path")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("source root is not a directory: " + str(root))
    for marker in project.markers:
        try:
            _contained_file(root, marker)
        except ValueError as exc:
            raise ValueError("invalid " + project.name + " source root: " + str(exc))
    return root


def source_manifest(project: Project, source_root: str) -> dict:
    """프로젝트에 등록된 각 진입점 파일의 소스 루트 내 가용성을 검사하여 매니페스트 딕셔너리를 반환합니다.

    루트를 검증한 뒤 각 항목의 절대 경로와 available/unavailable 상태 및 원인 사유를 수집합니다. 누락된 경로는 임의 대체되지 않으며 파일 실재 여부만 보고합니다."""
    root = validate_source_root(project, source_root)
    entries = []
    for entry in project.entries:
        try:
            path = _contained_file(root, entry.relative_path)
            status, reason = "available", ""
        except ValueError as exc:
            path = root / entry.relative_path
            status, reason = "unavailable", str(exc)
        entries.append(dict(name=entry.name, path=str(path),
                            reference_only=entry.reference_only,
                            status=status, reason=reason))
    return dict(project=project.name, source_root=str(root), entries=entries)


def build_launch_plan(project: Project, source_root: str, entry_name: str,
                      arguments: Sequence[str]) -> LaunchPlan:
    """허용된 진입점과 전달 인자를 검증하고 자식 프로세스 실행 계획(LaunchPlan)을 수립합니다.

    진입점이 project.entries에 정의되어 있고 arguments가 비어 있거나 str/bytes이면 거부합니다. 형식 계약은 문자열 시퀀스이며 모든 요소의 유형까지 검사하지는 않습니다. 소스 루트 내 스크립트 파일 경로를 고정하고 현재 인터프리터를 통한 실행 명령 튜플을 생성합니다."""
    entry = next((item for item in project.entries if item.name == entry_name), None)
    if entry is None:
        raise ValueError("entrypoint is not allowed: " + entry_name)
    if not arguments or isinstance(arguments, (str, bytes)):
        raise ValueError("explicit legacy arguments are required after --")
    root = validate_source_root(project, source_root)
    script = _contained_file(root, entry.relative_path)
    return LaunchPlan(root, (sys.executable, str(script), *arguments))


def execute_plan(plan: LaunchPlan) -> int:
    """LaunchPlan에 지정된 명령을 소스 루트를 작업 디렉터리(cwd)로 하여 자식 프로세스로 실행합니다.

    shell=False 환경에서 subprocess.run을 호출하며, 실행 종료 코드를 그대로 반환합니다. 상대 경로 인자는 plan.root 기준으로 해석됩니다."""
    return subprocess.run(list(plan.command), cwd=str(plan.root),
                          check=False).returncode
