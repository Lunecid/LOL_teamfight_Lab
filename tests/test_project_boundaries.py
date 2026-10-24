"""Small stdlib checks; no native dependencies, datasets, GPU, or training."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "project_boundaries", ROOT / "tools/check_project_boundaries.py")
boundaries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundaries)


class ProjectBoundariesTest(unittest.TestCase):
    def test_repository_boundaries(self):
        self.assertEqual(boundaries.check_project(ROOT), [])

    def test_packaging_rejects_legacy_discovery(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            metadata = tomllib.load(stream)
        for include in (["*"], ["c_ept", "train"], ["c_ept*"]):
            candidate = copy.deepcopy(metadata)
            candidate["tool"]["setuptools"]["packages"]["find"]["include"] = include
            self.assertTrue(boundaries.check_packaging(
                candidate, name="lol-teamfight-ept", package="c_ept",
                entry="c_ept.s4a_operator:main", where=".", subpackages=True))

    def test_rejects_command_imports_at_any_depth(self):
        sources = (
            "import c_ept.run_s3",
            "from c_ept import run_s3",
            "from . import train",
            "from ..run_s3 import main",
            "def f():\n    from c_ept.ept.train import fit",
            "importlib.import_module('c_ept.run_s4a')",
            "__import__('c_ept.ept.train')",
            "import_module('.train', __package__)",
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(boundaries.check_imports(source, "c_ept.ept.checkpoints"))
        self.assertEqual(boundaries.check_imports(
            "from c_ept.ept import training_step\nfrom pathlib import Path",
            "c_ept.ept.checkpoints"), [])

    def test_only_allowlisted_sources_are_opened(self):
        opened = []
        metadata_paths = set(boundaries.METADATA_PATHS)

        def read(path, *args, **kwargs):
            relative = path.relative_to(ROOT).as_posix()
            opened.append(relative)
            if relative in metadata_paths:
                return ""  # Packaging errors do not widen the scan.
            if relative in boundaries.SOURCE_MODULES:
                return "import os\n"
            raise AssertionError(f"Unapproved source read: {relative}")

        with mock.patch.object(Path, "read_text", read):
            boundaries.check_project(ROOT)
        self.assertEqual(set(opened), set(boundaries.SOURCE_MODULES) | metadata_paths)

    def test_shared_runtime_dependency_contract(self):
        metadata = tomllib.loads((ROOT / "COG/pyproject.toml").read_text())
        for dependencies in ([], ["lol-project-runtime"], ["lol-project-runtime==0.1.0", "torch"]):
            candidate = copy.deepcopy(metadata)
            candidate["project"]["dependencies"] = dependencies
            self.assertTrue(boundaries.check_packaging(
                candidate, name="lol-teamfight-cog", package="lol_teamfight_cog",
                entry="lol_teamfight_cog.cli:main", where="src",
                dependencies=["lol-project-runtime==0.1.0"]))

    def test_pure_modules_reject_research_dependencies(self):
        allowed = sys.stdlib_module_names | {"numpy"}
        for source in ("from studies.v_production_20261004 import train",
                       "import torch", "from core.config import cfg",
                       "from app.experiment import run"):
            with self.subTest(source=source):
                self.assertTrue(boundaries.check_imports(source, "pure.metrics", allowed_roots=allowed))
        self.assertEqual(boundaries.check_imports(
            "import numpy\nfrom typing import Any", "pure.metrics", allowed_roots=allowed), [])

    def test_launcher_runtime_rejects_scientific_imports(self):
        allowed = sys.stdlib_module_names | {"lol_project_runtime"}
        for source in ("import c_ept", "import torch", "from app import experiment",
                       "from lol_teamfight_master import cli"):
            with self.subTest(source=source):
                self.assertTrue(boundaries.check_imports(
                    source, "lol_project_runtime.launcher", allowed_roots=allowed))
        self.assertEqual(boundaries.check_imports(
            "from .launcher import Project", "lol_project_runtime.cli", allowed_roots=allowed), [])

    def test_console_target_help_in_fresh_process(self):
        code = """
import importlib, sys, tomllib
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
entry = tomllib.loads((root / 'pyproject.toml').read_text())['project']['scripts']['lol-teamfight-ept']
module, function = entry.split(':')
main = getattr(importlib.import_module(module), function)
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert not {'numpy', 'torch', 'scipy', 'c_ept.run_s3', 'c_ept.run_s4a', 'c_ept.ept.train'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT)],
                                cwd=ROOT / "tools", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--action", result.stdout)

    def test_default_check_never_imports_runner(self):
        from c_ept import s4a_operator as operator
        with contextlib.ExitStack() as stack:
            prepare = stack.enter_context(mock.patch.object(operator, "prepare_native_runtime", return_value={}))
            stack.enter_context(mock.patch.object(operator, "load_config", return_value={}))
            stack.enter_context(mock.patch.object(operator, "config_digest", return_value="mock"))
            preflight = stack.enter_context(mock.patch.object(operator, "preflight", return_value="mock"))
            importer = stack.enter_context(mock.patch.object(operator.importlib, "import_module", side_effect=AssertionError("runner import")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(operator.main([]), 0)
            prepare.assert_called_once_with()
            preflight.assert_called_once_with({})
            importer.assert_not_called()

    def test_cog_master_help_and_default_are_source_free(self):
        code = """
import importlib, sys
from pathlib import Path
from unittest import mock
root = Path(sys.argv[1])
sys.path[:0] = [str(root / p) for p in ('tools/project_runtime/src', 'COG/src', 'master_ver/src')]
with mock.patch('subprocess.run', side_effect=AssertionError('unexpected source execution')):
    for package in ('lol_teamfight_cog', 'lol_teamfight_master'):
        main = importlib.import_module(package + '.cli').main
        assert main([]) == 0
        try:
            main(['--help'])
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError('help did not exit')
assert not {'numpy', 'torch', 'c_ept', 'app', 'core', 'studies'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT)],
                                cwd=ROOT / "tools", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_check_refusal_does_not_train(self):
        from c_ept import s4a_operator as operator
        with mock.patch.object(operator, "prepare_native_runtime", return_value={}), \
             mock.patch.object(operator, "load_config", side_effect=operator.OperatorError("missing config")), \
             mock.patch.object(operator.importlib, "import_module", side_effect=AssertionError("runner import")) as importer, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(operator.main([]), 2)
            importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
