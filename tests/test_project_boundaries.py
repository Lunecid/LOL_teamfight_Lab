"""Small stdlib checks; no native dependencies, datasets, GPU, or training."""
import ast
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HELP_IMPORT_GUARD = """assert not {
    'numpy', 'torch', 'scipy', 'c_ept.run_s3', 'c_ept.run_s4a',
    'c_ept.workflows.run_s3', 'c_ept.workflows.run_s4a', 'c_ept.ept.train'
} & sys.modules.keys()
"""
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
""" + HELP_IMPORT_GUARD
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT)],
                                cwd=ROOT / "tools", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--action", result.stdout)

    def test_help_import_guard_rejects_each_forbidden_module_independently(self):
        for name in ('numpy', 'torch', 'scipy', 'c_ept.run_s3', 'c_ept.run_s4a',
                     'c_ept.workflows.run_s3', 'c_ept.workflows.run_s4a', 'c_ept.ept.train'):
            with self.subTest(module=name), self.assertRaises(AssertionError):
                exec(HELP_IMPORT_GUARD, {'sys': SimpleNamespace(modules={name: object()})})
        exec(HELP_IMPORT_GUARD, {'sys': SimpleNamespace(modules={'pathlib': object()})})

    def test_default_check_never_imports_runner(self):
        from c_ept.workflows import s4a_operator as operator
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
        from c_ept.workflows import s4a_operator as operator
        with mock.patch.object(operator, "prepare_native_runtime", return_value={}), \
             mock.patch.object(operator, "load_config", side_effect=operator.OperatorError("missing config")), \
             mock.patch.object(operator.importlib, "import_module", side_effect=AssertionError("runner import")) as importer, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(operator.main([]), 2)
            importer.assert_not_called()


def build_tree(base: Path, files: dict) -> Path:
    """Write a throw-away c_ept tree: {relative posix path: source}."""
    for relative, source in files.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return base


# A minimal package skeleton the rule tests share.
SKELETON = {
    "c_ept/__init__.py": "",
    "c_ept/run/__init__.py": "",
    "c_ept/run/foo.py": "def _helper():\n    pass\n",
    "c_ept/valuation/__init__.py": "",
    "c_ept/valuation/compose.py": "def _checked(x):\n    return x\n\ndef public(x):\n    return x\n",
    "c_ept/ept/__init__.py": "",
    "c_ept/ept/train.py": "class TrainConfig:\n    pass\n\n\ndef train():\n    pass\n",
    "c_ept/s4a_operator.py": "",
    "c_ept/tests/__init__.py": "",
    "c_ept/workflows/__init__.py": "",
    "c_ept/workflows/run_s9.py": "",
    "c_ept/workflows/s4a_operator.py": "",
    "c_ept/comparison/__init__.py": "",
    "c_ept/comparison/operator.py": "",
    "tools/x.py": "",
}
PRIVATE_COMPOSE = "c_ept.valuation.compose._checked"
PRIVATE_RUN = "c_ept.run.foo._helper"


def entry(importer, target, reason=None):
    result = {"importer": importer, "target": target}
    if reason is not None:
        result["reason"] = reason
    return result


class TreeBoundaryTest(unittest.TestCase):
    """D11/Q8: the whole c_ept tree, strict for c_ept/run and STRICT_FILES, shrink-only allowlist for the rest."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.allowlist = self.base / "boundary_allowlist.json"

    def write_allowlist(self, library=(), tests=(), frozen=(), white_box=()):
        self.allowlist.write_text(json.dumps({"version": 3, "library": list(library), "tests": list(tests),
                                              "frozen_reference": list(frozen), "white_box": list(white_box)}),
                                  encoding="utf-8")

    def errors(self, files, library=(), max_entries=None, tests=(), frozen=(), white_box=()):
        for relative, source in files.items():
            build_tree(self.base, {relative: source})
        build_tree(self.base, SKELETON)
        self.write_allowlist(library, tests, frozen, white_box)
        kwargs = {} if max_entries is None else {"max_entries": max_entries}
        return boundaries.check_tree(self.base, allowlist_path=self.allowlist, **kwargs)

    # --- strict rule for c_ept/run ---------------------------------------
    def test_run_package_underscore_import_fails(self):
        errors = self.errors({"c_ept/run/a.py": "from c_ept.valuation.compose import _checked\n"})
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("c_ept/run/a.py", errors[0])
        self.assertIn(PRIVATE_COMPOSE, errors[0])

    def test_run_package_underscore_attribute_and_module_forms_fail(self):
        for source in ("from c_ept.valuation import compose\nx = compose._checked\n",
                       "import c_ept.valuation.compose as cc\nx = cc._checked\n",
                       "import c_ept.valuation.compose\nx = c_ept.valuation.compose._checked\n",
                       "from ..valuation.compose import _checked\n",
                       "from c_ept.valuation import _hidden\n"):
            with self.subTest(source=source):
                errors = self.errors({"c_ept/run/a.py": source})
                self.assertEqual(len(errors), 1, errors)

    def test_literal_dynamic_import_targets_obey_the_private_rule(self):
        """I2: importlib.import_module / __import__ with a literal target is an import like any other."""
        for source in ("import importlib\nimportlib.import_module('c_ept.valuation._hidden')\n",
                       "import importlib\nx = importlib.import_module('c_ept.valuation.compose')._checked\n",
                       "x = __import__('c_ept.valuation._hidden')\n",
                       "from importlib import import_module\nimport_module('..valuation._hidden', __package__)\n",
                       "from importlib import import_module\nimport_module('c_ept.valuation._hidden.sub')\n"):
            with self.subTest(source=source):
                errors = self.errors({"c_ept/run/a.py": source})
                self.assertEqual(len(errors), 1, errors)
                self.assertRegex(errors[0], r"c_ept\.valuation\.(compose\.)?_\w+")
        # a non-literal argument cannot be resolved by a syntax check and is not claimed
        self.assertEqual(self.errors({"c_ept/run/a.py": "import importlib\nimportlib.import_module(name)\n"}), [])
        self.assertEqual(self.errors({"c_ept/run/a.py": "import importlib\nimportlib.import_module('c_ept.valuation.compose')\n"}), [])

    def test_target_is_cut_after_the_first_private_component(self):
        """m2: a trailing attribute past the private name is not a different violation."""
        errors = self.errors({"c_ept/run/a.py": "from c_ept.valuation import compose\ncompose._checked.cache_clear()\n"})
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(PRIVATE_COMPOSE, errors[0])
        self.assertNotIn("cache_clear", errors[0])

    def test_run_package_same_package_underscore_passes(self):
        errors = self.errors({"c_ept/run/b.py": "def _x():\n    pass\n",
                              "c_ept/run/a.py": "from c_ept.run.b import _x\nfrom .b import _x as y\n"})
        self.assertEqual(errors, [])

    def test_run_package_command_module_import_fails(self):
        for source in ("from c_ept.workflows import run_s9\n",
                       "import c_ept.workflows.run_s9\n",
                       "from c_ept.workflows.run_s9 import main\n",
                       "import tools.x\n",
                       "from tools import x\n",
                       "from c_ept.comparison import operator\n",
                       "from c_ept.comparison.operator import main\n",
                       "from ..workflows import run_s9\n",
                       "import c_ept.s4a_operator\n",
                       "from c_ept import s4a_operator\n",
                       "from c_ept.workflows import s4a_operator\n",
                       "import c_ept.workflows.s4a_operator\n",
                       "from ..workflows.s4a_operator import main\n",
                       "import importlib\nimportlib.import_module('c_ept.workflows.run_s9')\n"):
            with self.subTest(source=source):
                errors = self.errors({"c_ept/run/a.py": source})
                self.assertTrue(errors, source)
                self.assertTrue(all("c_ept/run/a.py" in e and "command module" in e for e in errors), errors)

    def test_run_package_may_import_the_training_loop(self):
        """I1: the command set is exactly run_*, tools, comparison.operator and s4a_operator; c_ept.ept.train is a library."""
        for source in ("from c_ept.ept.train import TrainConfig, train\n",
                       "import c_ept.ept.train\n",
                       "from c_ept.ept import train\n",
                       "from ..ept import train as t\n"):
            with self.subTest(source=source):
                self.assertEqual(self.errors({"c_ept/run/a.py": source}), [])

    def test_stdlib_and_dunder_underscore_passes(self):
        source = ("import _winapi\nfrom os import _exit\nfrom collections import _OrderedDictKeysView\n"
                  "import os\nx = os._exit\nfrom c_ept.valuation.compose import __doc__\n"
                  "from c_ept.valuation.compose import public\n")
        self.assertEqual(self.errors({"c_ept/run/a.py": source}), [])

    # --- allowlist for the rest -------------------------------------------
    def test_underscore_cross_package_outside_run_needs_allowlist(self):
        files = {"c_ept/ept/m.py": "from c_ept.valuation.compose import _checked\n"}
        errors = self.errors(files)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("c_ept/ept/m.py", errors[0])
        self.assertIn("not in the allowlist", errors[0])
        self.assertEqual(self.errors(files, [entry("c_ept/ept/m.py", PRIVATE_COMPOSE)]), [])

    def test_command_import_outside_run_is_not_a_tree_violation(self):
        self.assertEqual(self.errors({"c_ept/ept/m.py": "from c_ept.workflows import run_s9\n"}), [])

    def test_stale_entry_is_refused(self):
        errors = self.errors({"c_ept/ept/m.py": "x = 1\n"}, [entry("c_ept/ept/m.py", PRIVATE_COMPOSE)])
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("stale", errors[0])

    def test_grown_allowlist_is_refused_per_section(self):
        files = {"c_ept/ept/m.py": "from c_ept.valuation.compose import _checked\n",
                 "c_ept/ept/n.py": "from c_ept.valuation.compose import _checked\n",
                 "c_ept/tests/test_t.py": "from c_ept.valuation.compose import _checked\n"}
        library = [entry("c_ept/ept/m.py", PRIVATE_COMPOSE), entry("c_ept/ept/n.py", PRIVATE_COMPOSE)]
        tests = [entry("c_ept/tests/test_t.py", PRIVATE_COMPOSE)]
        limits = {"library": 2, "tests": 1, "frozen_reference": 0, "white_box": 0}
        self.assertEqual(self.errors(files, library, max_entries=limits, tests=tests), [])
        for section, count in (("library", 1), ("tests", 0)):
            with self.subTest(section=section):
                errors = self.errors(files, library, max_entries={**limits, section: count}, tests=tests)
                self.assertTrue(any("larger than the committed baseline" in e and section in e for e in errors), errors)

    def test_strict_run_file_cannot_be_allowlisted(self):
        files = {"c_ept/run/a.py": "from c_ept.valuation.compose import _checked\n"}
        for kwargs in ({"library": [entry("c_ept/run/a.py", PRIVATE_COMPOSE)]},
                       {"tests": [entry("c_ept/run/a.py", PRIVATE_COMPOSE)]},
                       {"frozen": [entry("c_ept/run/a.py", PRIVATE_COMPOSE, "why")]}):
            with self.subTest(section=list(kwargs)):
                errors = self.errors(files, **kwargs)
                self.assertTrue(any("strict" in e for e in errors), errors)
                self.assertTrue(any("c_ept/run/a.py:1" in e for e in errors), errors)

    # --- STRICT_FILES and the frozen_reference section (I3) -----------------
    STRICT_TEST = "c_ept/tests/test_run_new.py"
    STRICT_SOURCE = ("from c_ept.valuation.compose import _checked\n"
                     "from c_ept.workflows import run_s9\n")

    def strict_errors(self, **kwargs):
        with mock.patch.object(boundaries, "STRICT_FILES", frozenset({self.STRICT_TEST})):
            return self.errors({self.STRICT_TEST: self.STRICT_SOURCE}, **kwargs)

    def test_strict_files_apply_both_halves_of_the_strict_rule(self):
        errors = self.strict_errors()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(any("private import" in e and PRIVATE_COMPOSE in e for e in errors), errors)
        self.assertTrue(any("command module" in e and "c_ept.workflows.run_s9" in e for e in errors), errors)
        # the same file outside STRICT_FILES: only the private half, via the tests section
        errors = self.errors({self.STRICT_TEST: self.STRICT_SOURCE})
        self.assertEqual(len(errors), 1, errors)
        self.assertEqual(self.errors({self.STRICT_TEST: self.STRICT_SOURCE},
                                     tests=[entry(self.STRICT_TEST, PRIVATE_COMPOSE)]), [])

    def test_frozen_reference_holds_strict_test_files_with_a_reason(self):
        frozen = [entry(self.STRICT_TEST, "c_ept.workflows.run_s9", "frozen command module"),
                  entry(self.STRICT_TEST, PRIVATE_COMPOSE, "frozen private function")]
        frozen.sort(key=lambda e: (e["importer"], e["target"]))
        self.assertEqual(self.strict_errors(frozen=frozen), [])
        # a stale frozen entry is refused like any other
        extra = sorted(frozen + [entry(self.STRICT_TEST, "c_ept.valuation.compose._other", "r")], key=lambda e: (e["importer"], e["target"]))
        errors = self.strict_errors(frozen=extra)
        self.assertTrue(any("stale" in e for e in errors), errors)

    def test_frozen_reference_is_shrink_only_and_only_for_strict_test_files(self):
        frozen = [entry(self.STRICT_TEST, "c_ept.workflows.run_s9", "r"), entry(self.STRICT_TEST, PRIVATE_COMPOSE, "r")]
        frozen.sort(key=lambda e: (e["importer"], e["target"]))
        errors = self.strict_errors(frozen=frozen, max_entries={"library": 0, "tests": 0, "frozen_reference": 1, "white_box": 0})
        self.assertTrue(any("larger than the committed baseline" in e and "frozen_reference" in e for e in errors), errors)
        # not a strict file: refused by section membership
        other = {"c_ept/tests/test_other.py": "from c_ept.valuation.compose import _checked\n"}
        errors = self.errors(other, frozen=[entry("c_ept/tests/test_other.py", PRIVATE_COMPOSE, "r")])
        self.assertTrue(any("frozen_reference" in e for e in errors), errors)
        # a library module cannot be frozen
        errors = self.errors({"c_ept/ept/m.py": "from c_ept.valuation.compose import _checked\n"},
                             frozen=[entry("c_ept/ept/m.py", PRIVATE_COMPOSE, "r")])
        self.assertTrue(any("frozen_reference" in e for e in errors), errors)

    WHITE_BOX_SOURCE = "from c_ept.run import foo\nfoo._helper()\n"

    def white_box_errors(self, source=None, **kwargs):
        with mock.patch.object(boundaries, "STRICT_FILES", frozenset({self.STRICT_TEST})):
            return self.errors({self.STRICT_TEST: source or self.WHITE_BOX_SOURCE}, **kwargs)

    def test_white_box_holds_a_strict_test_files_private_run_seam(self):
        """A kernel test may reach a private name inside c_ept.run.*, listed with a reason in white_box only."""
        errors = self.white_box_errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(PRIVATE_RUN, errors[0])
        listed = [entry(self.STRICT_TEST, PRIVATE_RUN, "white-box seam of the kernel's own module")]
        self.assertEqual(self.white_box_errors(white_box=listed), [])
        # the private-name target is cut after the private component, so a deeper use is the same entry
        deeper = "from c_ept.run import foo\nfoo._helper.cache_clear()\n"
        self.assertEqual(self.white_box_errors(deeper, white_box=listed), [])
        # a stale white_box entry is refused like any other
        errors = self.white_box_errors("x = 1\n", white_box=listed)
        self.assertTrue(any("stale" in e for e in errors), errors)

    def test_white_box_is_narrow(self):
        reason = "white-box seam of the kernel's own module"
        # target outside c_ept.run.*: not a kernel seam
        other = "from c_ept.valuation.compose import _checked\n"
        errors = self.white_box_errors(other, white_box=[entry(self.STRICT_TEST, PRIVATE_COMPOSE, reason)])
        self.assertTrue(any("white_box" in e for e in errors), errors)
        self.assertTrue(any("not in the allowlist" in e or "strict rule" in e for e in errors), errors)
        # a non-strict test file cannot use it
        files = {"c_ept/tests/test_other.py": self.WHITE_BOX_SOURCE}
        errors = self.errors(files, white_box=[entry("c_ept/tests/test_other.py", PRIVATE_RUN, reason)])
        self.assertTrue(any("white_box" in e for e in errors), errors)
        # c_ept/run itself can never be listed, in any section
        files = {"c_ept/run/a.py": "from c_ept.run import foo\nfrom c_ept.valuation.compose import _checked\n"}
        errors = self.errors(files, white_box=[entry("c_ept/run/a.py", PRIVATE_COMPOSE, reason)])
        self.assertTrue(any("strict" in e and "c_ept/run/a.py" in e for e in errors), errors)
        # a library module cannot use it
        files = {"c_ept/ept/m.py": "from c_ept.run import foo\nfoo._helper()\n"}
        errors = self.errors(files, white_box=[entry("c_ept/ept/m.py", PRIVATE_RUN, reason)])
        self.assertTrue(any("white_box" in e for e in errors), errors)

    def test_frozen_reference_does_not_hold_run_seams(self):
        listed = [entry(self.STRICT_TEST, PRIVATE_RUN, "frozen reference")]
        errors = self.white_box_errors(frozen=listed)
        self.assertTrue(any("frozen_reference" in e for e in errors), errors)

    def test_white_box_is_shrink_only(self):
        listed = [entry(self.STRICT_TEST, PRIVATE_RUN, "white-box seam of the kernel's own module")]
        errors = self.white_box_errors(white_box=listed,
                                       max_entries={"library": 0, "tests": 0, "frozen_reference": 0, "white_box": 0})
        self.assertTrue(any("larger than the committed baseline" in e and "white_box" in e for e in errors), errors)

    def test_section_membership_is_enforced(self):
        files = {"c_ept/tests/test_t.py": "from c_ept.valuation.compose import _checked\n",
                 "c_ept/ept/m.py": "from c_ept.valuation.compose import _checked\n"}
        errors = self.errors(files, library=[entry("c_ept/ept/m.py", PRIVATE_COMPOSE),
                                             entry("c_ept/tests/test_t.py", PRIVATE_COMPOSE)])
        self.assertTrue(any("library" in e and "c_ept/tests/test_t.py" in e for e in errors), errors)
        errors = self.errors(files, tests=[entry("c_ept/ept/m.py", PRIVATE_COMPOSE),
                                           entry("c_ept/tests/test_t.py", PRIVATE_COMPOSE)])
        self.assertTrue(any("tests" in e and "c_ept/ept/m.py" in e for e in errors), errors)

    def test_allowlist_is_strict_json(self):
        one = entry("c_ept/ept/m.py", "c_ept.a._b")
        two = entry("c_ept/ept/n.py", "c_ept.a._b")

        def doc(**sections):
            body = {"version": 3, "library": [], "tests": [], "frozen_reference": [], "white_box": []}
            body.update(sections)
            return json.dumps(body)

        bad = {
            "duplicate key": '{"version": 3, "version": 3, "library": [], "tests": [], "frozen_reference": [], "white_box": []}',
            "unsorted": doc(library=[two, one]),
            "duplicate entry": doc(library=[one, one]),
            "line number": doc(library=[{**one, "line": 3}]),
            "reason outside frozen": doc(library=[{**one, "reason": "x"}]),
            "frozen without reason": doc(frozen_reference=[entry("c_ept/tests/test_run_io.py", "c_ept.a._b")]),
            "dot-dot importer": doc(library=[entry("c_ept/../x.py", "c_ept.a._b")]),
            "importer outside c_ept": doc(library=[entry("tools/x.py", "c_ept.a._b")]),
            "bad target": doc(library=[entry("c_ept/ept/m.py", "c_ept")]),
            "white_box without reason": doc(white_box=[entry("c_ept/tests/test_run_io.py", "c_ept.run.foo._h")]),
            "missing section": json.dumps({"version": 3, "library": [], "tests": [], "frozen_reference": []}),
            "extra section": doc(extra=[]),
            "not json": "[",
            "old version": json.dumps({"version": 2, "library": [], "tests": [], "frozen_reference": []}),
            "nan": '{"version": 3, "library": [], "tests": [], "frozen_reference": [], "white_box": [], "x": NaN}',
        }
        for label, text in bad.items():
            with self.subTest(label=label):
                build_tree(self.base, SKELETON)
                self.allowlist.write_text(text, encoding="utf-8")
                errors = boundaries.check_tree(self.base, allowlist_path=self.allowlist)
                self.assertTrue(any("allowlist" in e for e in errors), errors)

    def test_write_allowlist_is_sorted_and_round_trips(self):
        sections = {
            "library": [("c_ept/ept/n.py", "c_ept.a._b"), ("c_ept/ept/m.py", "c_ept.z._y"),
                        ("c_ept/ept/m.py", "c_ept.a._b")],
            "tests": [("c_ept/tests/test_b.py", "c_ept.a._b"), ("c_ept/tests/test_a.py", "c_ept.a._b")],
            "frozen_reference": [("c_ept/tests/test_run_io.py", "c_ept.a._b", "why")],
            "white_box": [("c_ept/tests/test_run_io.py", "c_ept.run.z._b", "why"),
                          ("c_ept/tests/test_run_io.py", "c_ept.run.a._b", "why")],
        }
        boundaries.write_allowlist(self.allowlist, sections)
        loaded = boundaries.load_allowlist(self.allowlist)
        self.assertEqual(loaded, {name: sorted(items) for name, items in sections.items()})
        document = json.loads(self.allowlist.read_text(encoding="utf-8"))
        self.assertEqual(document["version"], 3)
        self.assertEqual(set(document["library"][0]), {"importer", "target"})
        self.assertEqual(set(document["frozen_reference"][0]), {"importer", "target", "reason"})
        self.assertEqual(set(document["white_box"][0]), {"importer", "target", "reason"})

    # --- protected files ---------------------------------------------------
    def test_protected_paths_are_excluded_by_name(self):
        for relative in ("c_ept/cohort.py", "research_pipeline/cohort_policy.py",
                         "research_pipeline/holdout_r3/x.py", "research_pipeline/holdout_r3/sub/y.py"):
            self.assertTrue(boundaries.is_protected(relative), relative)
        for relative in ("c_ept/config.py", "c_ept/cohort_x.py", "c_ept/ept/cohort.py",
                         "research_pipeline/other.py"):
            self.assertFalse(boundaries.is_protected(relative), relative)

    def test_protected_files_are_never_opened_or_parsed(self):
        build_tree(self.base, SKELETON)
        # Poison: unparsable, and an underscore import that would be reported if it were read.
        build_tree(self.base, {"c_ept/cohort.py": "POISON(\nfrom c_ept.valuation.compose import _checked\n",
                               "c_ept/ept/cohort.py": "x = 1\n"})
        self.write_allowlist()
        touched, parsed = [], []
        real_read, real_open, real_parse = Path.read_text, builtins.open, ast.parse

        def spy_read(path, *args, **kwargs):
            touched.append(Path(path).as_posix())
            return real_read(path, *args, **kwargs)

        def spy_open(file, *args, **kwargs):
            touched.append(Path(str(file)).as_posix())
            return real_open(file, *args, **kwargs)

        def spy_parse(source, filename="<unknown>", *args, **kwargs):
            parsed.append(str(filename))
            return real_parse(source, filename, *args, **kwargs)

        with mock.patch.object(Path, "read_text", spy_read), \
             mock.patch.object(builtins, "open", spy_open), \
             mock.patch.object(ast, "parse", spy_parse):
            errors = boundaries.check_tree(self.base, allowlist_path=self.allowlist)
        self.assertEqual(errors, [])
        base = self.base.as_posix()
        self.assertNotIn(f"{base}/c_ept/cohort.py", touched)
        self.assertFalse([p for p in parsed if p.replace("\\", "/").endswith("c_ept/cohort.py")], parsed)
        self.assertIn(f"{base}/c_ept/ept/cohort.py", touched)  # only the exact path is excluded
        self.assertEqual(boundaries.walk_package(self.base).count("c_ept/cohort.py"), 0)

    def make_directory_link(self, link: Path, target: Path) -> bool:
        """A symlink where the host allows it, otherwise a Windows junction; False if neither can be made."""
        try:
            os.symlink(target, link, target_is_directory=True)
            return True
        except (OSError, NotImplementedError, AttributeError):
            pass
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True, text=True)
            return result.returncode == 0
        return False

    def test_links_under_c_ept_are_refused_and_never_followed(self):
        """m1: a protected path reached through a link would otherwise be read under a harmless name."""
        build_tree(self.base, SKELETON)
        outside = self.base / "outside"
        build_tree(outside, {"leak.py": "POISON(\n"})
        if not self.make_directory_link(self.base / "c_ept" / "linked", outside):
            self.skipTest("this host can create neither a symlink nor a junction")
        self.write_allowlist()
        touched = []
        real_read = Path.read_text

        def spy_read(path, *args, **kwargs):
            touched.append(Path(path).as_posix())
            return real_read(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", spy_read):
            errors = boundaries.check_tree(self.base, allowlist_path=self.allowlist)
        self.assertTrue(any("link" in e and "c_ept/linked" in e for e in errors), errors)
        self.assertFalse([p for p in touched if p.endswith("leak.py")], touched)
        self.assertNotIn("c_ept/linked/leak.py", boundaries.walk_package(self.base))

    def test_file_symlink_under_c_ept_is_refused(self):
        build_tree(self.base, SKELETON)
        build_tree(self.base, {"elsewhere.py": "x = 1\n"})
        try:
            os.symlink(self.base / "elsewhere.py", self.base / "c_ept" / "ept" / "alias.py")
        except (OSError, NotImplementedError):
            self.skipTest("this host cannot create file symlinks")
        self.write_allowlist()
        errors = boundaries.check_tree(self.base, allowlist_path=self.allowlist)
        self.assertTrue(any("link" in e and "c_ept/ept/alias.py" in e for e in errors), errors)
        self.assertNotIn("c_ept/ept/alias.py", boundaries.walk_package(self.base))

    # --- the real tree -----------------------------------------------------
    def test_real_tree_passes_and_never_touches_protected_files(self):
        touched = []
        real_read = Path.read_text

        def spy_read(path, *args, **kwargs):
            touched.append(Path(path).relative_to(ROOT).as_posix())
            return real_read(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", spy_read):
            self.assertEqual(boundaries.check_tree(ROOT), [])
        self.assertTrue(touched)
        self.assertEqual([p for p in touched if boundaries.is_protected(p)], [])
        walked = boundaries.walk_package(ROOT)
        self.assertGreater(len(walked), 50)
        self.assertIn("c_ept/run/supervisor.py", walked)

    def test_strict_files_are_real_and_cover_the_s4b_prep_tests(self):
        walked = set(boundaries.walk_package(ROOT))
        self.assertTrue(boundaries.STRICT_FILES <= walked, sorted(boundaries.STRICT_FILES - walked))
        for name in ("identity", "ledger", "supervisor", "guards", "io", "profile", "runners", "evaluate", "gate0", "numeric", "measure", "cli"):
            self.assertIn(f"c_ept/tests/test_run_{name}.py", boundaries.STRICT_FILES)
        self.assertIn("c_ept/tests/run_runner_bodies.py", boundaries.STRICT_FILES)             # the loader and stage bodies of the runner tests (S4b-prep K6a)
        self.assertIn("c_ept/tests/run_evaluate_bodies.py", boundaries.STRICT_FILES)           # the loader and the backend of the evaluation-runner tests (S4b-prep K6b)
        self.assertIn("c_ept/tests/run_gate0_bodies.py", boundaries.STRICT_FILES)              # the store builder of the Gate 0 supervised tests (EPT-T013)
        self.assertIn("c_ept/tests/run_measure_bodies.py", boundaries.STRICT_FILES)            # the stage bodies of the measurement-runner tests (S4b-prep K6c)
        run_modules = {p for p in walked if p.startswith("c_ept/run/")}
        self.assertTrue(run_modules and all(boundaries.is_strict(p) for p in run_modules))

    def test_committed_allowlist_is_pinned_per_section(self):
        """Each section's count and digest only ever move down together with the file and ALLOWLIST_MAX_ENTRIES."""
        sections = boundaries.load_allowlist(ROOT / "tools/boundary_allowlist.json")
        self.assertEqual(set(sections), set(PINNED_ALLOWLIST))
        for name, (count, digest) in PINNED_ALLOWLIST.items():
            with self.subTest(section=name):
                entries = sections[name]
                self.assertEqual(len(entries), count)
                self.assertEqual(boundaries.ALLOWLIST_MAX_ENTRIES[name], count)
                canonical = "\n".join(f"{item[0]}\t{item[1]}" for item in entries)
                self.assertEqual(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), digest)
                self.assertEqual(sorted(entries), entries)
                self.assertFalse([item for item in entries if item[0].startswith("c_ept/run/")])
        self.assertTrue(all(item[0].startswith("c_ept/tests/") for item in sections["tests"]))
        self.assertTrue(all(item[0] in boundaries.STRICT_FILES and item[2].strip()
                            and not item[1].startswith("c_ept.run.")
                            for item in sections["frozen_reference"]))
        self.assertTrue(all(item[0] in boundaries.STRICT_FILES and item[2].strip()
                            and item[1].startswith("c_ept.run.")
                            for item in sections["white_box"]))

    def test_main_reports_the_tree_check(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(boundaries.main(), 0)
        self.assertIn("c_ept modules", out.getvalue())


# Baselines of the committed allowlist sections (K7). When a violation is fixed, delete its
# entry, lower ALLOWLIST_MAX_ENTRIES and these pins together; they never move up -- except once,
# after the merge of the two tracks (lead ruling INT1: integration baseline after the D25 merge,
# shrink-only from here; recorded in the X1 block of the S4b-prep plan): "tests" went
# from 101 to 109 for the eight private imports that the Track M white-box test files carry
# (test_ept_train, test_kcondition, test_killlaw; reason "S4b-prep Track M white-box test,
# integration baseline D25"). library, frozen_reference and white_box did not change. From this
# baseline every section only shrinks.
PINNED_ALLOWLIST = {
    "library": (35, "b804b6b3ef25d1fc5c2ad7404dcbdffcfce054d8ca2693270a3793aad77438b9"),
    "tests": (109, "1eebb4bfe1cd9951f06ab615481f6d41f258f7aa3c5c97ea0e2e32381a69f94e"),
    "frozen_reference": (3, "066b6952db8293ede3f226cec505b6bf17573d5ae8a176ae32c965ed8a3b45ba"),
    "white_box": (9, "a21da03368f3f5b184c14da4171be96cfa1396aa6539fa03c0bff757976881ae"),
}


if __name__ == "__main__":
    unittest.main()
