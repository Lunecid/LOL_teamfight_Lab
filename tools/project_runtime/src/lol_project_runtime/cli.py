"""연구 코드를 로드하지 않고 프로젝트 진입점 검사 및 실행 명령을 위임하는 CLI 진입 모듈.

list, check, run 명령을 제공하며, -- 구분자를 통해 연구 스크립트로 전달되는 레거시 인자를 안전하게 분리 파싱합니다."""
import argparse
import json
import sys
from typing import Optional, Sequence

from .launcher import (Project, build_launch_plan, execute_plan,
                       source_manifest)


def build_parser(project: Project) -> argparse.ArgumentParser:
    """실행은 run 하위 명령으로만 허용하고 소스 루트를 항상 요구한다."""
    parser = argparse.ArgumentParser(
        description=project.name + " source-backed launcher (research sources required)")
    commands = parser.add_subparsers(dest="action")
    for action in ("list", "check", "run"):
        command = commands.add_parser(action)
        command.add_argument("--source-root", required=True,
                             help="absolute path to this project's source tree")
        if action != "list":
            command.add_argument("--entrypoint", required=True,
                                 choices=[entry.name for entry in project.entries])
    return parser


def main(project: Project, argv: Optional[Sequence[str]] = None) -> int:
    """프로젝트 CLI 인터페이스 진입점으로 서브커맨드(list, check, run)를 파싱 및 디스패치합니다.

    인자 없이 호출되면 도움말을 출력하고 0을 반환합니다. -- 이후 인자는 run 하위 명령에만 허용됩니다. check 대상이 이용 불가하거나 검증 예외(ValueError, OSError) 발생 시 오류 메시지 출력 또는 비정상 종료 코드를 반환합니다.
    check의 파일 누락은 반환값 2, 인자/경로 오류는 argparse의 SystemExit(2)입니다. run은 자식 종료 코드를 반환합니다. 도움말/점검은 연구를 실행하지 않습니다."""
    parser = build_parser(project)
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        parser.print_help()
        return 0
    separator = tokens.index("--") if "--" in tokens else len(tokens)
    options = parser.parse_args(tokens[:separator])
    forwarded = tokens[separator + 1:]
    if options.action is None:
        parser.error("choose list, check or run")
    if options.action != "run" and separator != len(tokens):
        parser.error("legacy arguments are only accepted by run")
    try:
        if options.action == "list":
            print(json.dumps(source_manifest(project, options.source_root), indent=2))
            return 0
        if options.action == "check":
            manifest = source_manifest(project, options.source_root)
            entry = next(item for item in manifest["entries"]
                         if item["name"] == options.entrypoint)
            print(json.dumps(entry, indent=2))
            return 0 if entry["status"] == "available" else 2
        plan = build_launch_plan(project, options.source_root,
                                 options.entrypoint, forwarded)
        return execute_plan(plan)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2
