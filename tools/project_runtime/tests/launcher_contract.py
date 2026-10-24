"""합성 소스와 mocked subprocess만 사용한다. 연구 모듈/데이터는 읽지 않는다."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lol_project_runtime.launcher import (LaunchPlan, build_launch_plan, execute_plan,
                                         _contained_file, source_manifest, validate_source_root)


class LauncherContract:
    """프로젝트 설정별로 같은 실행 계약을 검사하는 mixin; 과학 구현은 호출하지 않는다."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="launcher sources ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for relative in self.project.markers:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("raise AssertionError('never import synthetic sources')",
                              encoding="utf-8")
        for entry in self.project.entries:
            if not entry.reference_only:
                target = self.root / entry.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("raise AssertionError('never execute sources')",
                                  encoding="utf-8")
        self.entry = self.project.entries[0]

    def call_cli(self, tokens):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = self.cli.main(tokens)
        return code, output.getvalue()

    def test_empty_command_prints_help_without_execution(self):
        with patch("subprocess.run") as child:
            code, output = self.call_cli([])
        self.assertEqual(code, 0)
        self.assertIn("source-backed", output)
        child.assert_not_called()

    def test_help_never_executes_or_requires_root(self):
        with patch("subprocess.run") as child, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.cli.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        child.assert_not_called()

    def test_root_is_explicit_and_never_inferred_from_cwd(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            validate_source_root(self.project, ".")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            validate_source_root(self.project, str(self.root / "missing"))

    def test_wrong_project_root_fails(self):
        (self.root / self.project.markers[0]).unlink()
        with self.assertRaisesRegex(ValueError, "invalid .* source root"):
            validate_source_root(self.project, str(self.root))

    def test_unknown_entry_is_not_executable(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            build_launch_plan(self.project, str(self.root), "../arbitrary.py", ["--help"])

    def test_empty_or_string_arguments_fail_closed(self):
        for arguments in ([], "--help"):
            with self.assertRaisesRegex(ValueError, "explicit legacy arguments"):
                build_launch_plan(self.project, str(self.root), self.entry.name, arguments)

    def test_plan_uses_external_absolute_root_and_preserves_arguments(self):
        arguments = ["--output", "relative output", "--source-root", "child-value", "--"]
        plan = build_launch_plan(self.project, str(self.root), self.entry.name, arguments)
        self.assertEqual(plan.root, self.root)
        self.assertEqual(plan.command, (sys.executable,
                         str(self.root / self.entry.relative_path), *arguments))

    def test_execute_preserves_exit_status_and_uses_no_shell(self):
        plan = LaunchPlan(self.root, (sys.executable, "synthetic script.py", "a b"))
        with patch("subprocess.run") as child:
            child.return_value.returncode = 23
            self.assertEqual(execute_plan(plan), 23)
            child.assert_called_once_with(list(plan.command), cwd=str(self.root), check=False)

    def test_cli_forwards_args_and_failure_exit_without_import(self):
        arguments = ["--output", "relative output", "--mode", "synthetic"]
        with patch("subprocess.run") as child:
            child.return_value.returncode = 17
            code, _ = self.call_cli(["run", "--source-root", str(self.root),
                                    "--entrypoint", self.entry.name, "--", *arguments])
        self.assertEqual(code, 17)
        self.assertEqual(child.call_args.args[0][2:], arguments)
        self.assertEqual(child.call_args.kwargs["cwd"], str(self.root))

    def test_run_without_explicit_arguments_never_executes(self):
        with patch("subprocess.run") as child, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.cli.main(["run", "--source-root", str(self.root),
                          "--entrypoint", self.entry.name])
        self.assertEqual(error.exception.code, 2)
        child.assert_not_called()

    def test_list_and_check_are_file_only_preflights(self):
        before = {relative: (self.root / relative).read_bytes()
                  for relative in self.project.markers}
        with patch("subprocess.run") as child:
            code, output = self.call_cli(["list", "--source-root", str(self.root)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["project"], self.project.name)
            code, output = self.call_cli(["check", "--source-root", str(self.root),
                                         "--entrypoint", self.entry.name])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["status"], "available")
        child.assert_not_called()
        self.assertEqual(before, {relative: (self.root / relative).read_bytes()
                                  for relative in self.project.markers})

    def test_check_rejects_forwarded_arguments(self):
        with patch("subprocess.run") as child, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.cli.main(["check", "--source-root", str(self.root),
                          "--entrypoint", self.entry.name, "--", "--help"])
        child.assert_not_called()

    def test_missing_entry_does_not_fallback(self):
        (self.root / self.entry.relative_path).unlink()
        with patch("subprocess.run") as child:
            with self.assertRaises(ValueError):
                plan = build_launch_plan(self.project, str(self.root), self.entry.name, ["--help"])
                execute_plan(plan)
        child.assert_not_called()

    def test_reference_only_entry_is_reported_and_fails_clearly(self):
        references = [entry for entry in self.project.entries if entry.reference_only]
        if not references:
            self.skipTest("this project has no reference-only entrypoint")
        entry = references[0]
        with patch("subprocess.run") as child:
            code, output = self.call_cli(["check", "--source-root", str(self.root),
                                         "--entrypoint", entry.name])
            self.assertEqual(code, 2)
            self.assertIn("source file unavailable", json.loads(output)["reason"])
            with self.assertRaisesRegex(ValueError, "source file unavailable"):
                build_launch_plan(self.project, str(self.root), entry.name, ["--help"])
        child.assert_not_called()

    def test_process_start_failure_is_actionable(self):
        with patch("subprocess.run", side_effect=OSError("synthetic start failure")):
            with contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as error:
                    self.cli.main(["run", "--source-root", str(self.root),
                              "--entrypoint", self.entry.name, "--", "--help"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("synthetic start failure", output.getvalue())

    def test_marker_symlink_cannot_escape_root(self):
        target = self.root / self.project.markers[0]
        target.unlink()
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external) / "marker"
            outside.write_text("synthetic", encoding="utf-8")
            try:
                target.symlink_to(outside)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            with self.assertRaisesRegex(ValueError, "escapes root"):
                validate_source_root(self.project, str(self.root))

    def test_resolved_path_escape_is_rejected_without_symlink_privileges(self):
        outside = self.root.parent / "synthetic outside.py"
        with patch.object(Path, "resolve", return_value=outside):
            with self.assertRaisesRegex(ValueError, "escapes root"):
                _contained_file(self.root, "synthetic link.py")

    def test_external_reference_can_be_used_when_explicitly_present(self):
        references = [entry for entry in self.project.entries if entry.reference_only]
        if not references:
            self.skipTest("this project has no reference-only entrypoint")
        entry = references[0]
        target = self.root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("raise AssertionError('never execute reference')", encoding="utf-8")
        with patch("subprocess.run") as child:
            plan = build_launch_plan(self.project, str(self.root), entry.name, ["--help"])
            self.assertEqual(plan.command[1], str(target))
            records = source_manifest(self.project, str(self.root))["entries"]
            self.assertEqual(next(item for item in records if item["name"] == entry.name)
                             ["status"], "available")
        child.assert_not_called()

    def test_packaging_exports_only_unique_launcher_namespace(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("metadata assertion requires Python 3.11+")
        metadata = tomllib.loads((self.project_root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["name"], self.distribution)
        self.assertEqual(metadata["project"]["dependencies"], ["lol-project-runtime==0.1.0"])
        finder = metadata["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(finder["where"], ["src"])
        self.assertEqual(finder["include"], [self.namespace])
        self.assertFalse(finder["namespaces"])
        self.assertEqual(metadata["tool"]["setuptools"]["py-modules"], [])
        self.assertIn(self.distribution + "[legacy,dev,analysis,acquisition]",
                      metadata["project"]["optional-dependencies"]["all"])
