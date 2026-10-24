"""공용 구현 연결/배포 경계만 검사한다. 과학 코드나 subprocess는 실행하지 않는다."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = RUNTIME_ROOT.parents[1]
for source in (RUNTIME_ROOT / "src", REPOSITORY / "COG/src",
               REPOSITORY / "master_ver/src"):
    sys.path.insert(0, str(source))
from lol_project_runtime import cli as shared_cli
from lol_project_runtime.launcher import Project
from lol_teamfight_cog import cli as cog_cli
from lol_teamfight_master import cli as master_cli


class RuntimeTests(unittest.TestCase):
    def test_adapters_use_one_cli_and_configuration_type(self):
        for adapter in (cog_cli, master_cli):
            self.assertIs(adapter.runtime_main, shared_cli.main)
            self.assertIsInstance(adapter.PROJECT, Project)
        self.assertNotEqual(cog_cli.PROJECT, master_cli.PROJECT)

    def test_every_adapter_subcommand_help_is_safe(self):
        with patch("subprocess.run") as child:
            for adapter in (cog_cli, master_cli):
                for command in ("list", "check", "run"):
                    with self.subTest(project=adapter.PROJECT.name, command=command):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as error:
                                adapter.main([command, "--help"])
                        self.assertEqual(error.exception.code, 0)
        child.assert_not_called()

    def test_shared_distribution_has_no_research_dependencies(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("metadata assertion requires Python 3.11+")
        metadata = tomllib.loads((RUNTIME_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["name"], "lol-project-runtime")
        self.assertEqual(metadata["project"]["version"], "0.1.0")
        self.assertEqual(metadata["project"]["dependencies"], [])
        finder = metadata["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(finder["include"], ["lol_project_runtime"])
        self.assertFalse(finder["namespaces"])

    def test_distributions_discover_disjoint_unique_namespaces(self):
        from setuptools import find_packages
        expected = ((RUNTIME_ROOT, "lol_project_runtime"),
                    (REPOSITORY / "COG", "lol_teamfight_cog"),
                    (REPOSITORY / "master_ver", "lol_teamfight_master"))
        for root, namespace in expected:
            self.assertEqual(find_packages(str(root / "src")), [namespace])


if __name__ == "__main__":
    unittest.main()
