"""Organization checks only: never execute a real report's main function."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "master_ver"
BASELINE = "d9d7ac8dbd5c389c8bf70cbd4c0ed1ee8be99e4a"
NAMES = (
    "report_b_three_horizons", "report_causal_position",
    "report_event_boundary_cif", "report_fixed120_events",
    "report_position_ablation", "report_postkill_full",
    "report_validation_suite", "report_window_validation",
)
MAPPING = {f"scripts/{name}.py": f"scripts/reporting/{name}.py" for name in NAMES}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
}


def dump(node):
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        if "type_params" in item._fields and not getattr(item, "type_params", None):
            item._fields = tuple(field for field in item._fields if field != "type_params")
    return ast.dump(node, include_attributes=False)


def fingerprint(node):
    return hashlib.sha256(dump(node).encode("utf-8")).hexdigest()


def added_inventory():
    return ast.parse(
        "files += [" + ", ".join("ROOT / " + repr(path) for path in MAPPING.values()) + "]"
    ).body[0]


def normalized_canonical(tree, name):
    """Accept precisely one ROOT relocation and the one exact inventory addition."""
    tree = copy.deepcopy(tree)
    roots = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "ROOT" for t in node.targets)]
    if len(roots) != 1 or dump(roots[0].value) != dump(ast.parse(
        "Path(__file__).resolve().parents[2]", mode="eval").body):
        raise AssertionError("unexpected ROOT expression")
    roots[0].value = ast.parse("Path(__file__).resolve().parents[1]", mode="eval").body
    if name == "report_validation_suite":
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        matches = [i for i, node in enumerate(main.body) if dump(node) == dump(added_inventory())]
        if len(matches) != 1:
            raise AssertionError("expected exactly the eight canonical provenance paths")
        index = matches[0]
        previous = main.body[index - 1]
        if not (isinstance(previous, ast.Assign) and len(previous.targets) == 1
                and isinstance(previous.targets[0], ast.Name)
                and previous.targets[0].id == "files"):
            raise AssertionError("inventory addition must follow the existing files assignment")
        del main.body[index]
    return tree


# Child imports exercise actual wrappers/implementations with inert scientific
# dependencies. Any unguarded report I/O would fail, rather than regenerate data.
IMPORT_CHECK = r'''
import importlib, json, pathlib, sys, types
for name in ("numpy", "pandas", "matplotlib", "matplotlib.pyplot", "sklearn", "sklearn.metrics"):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
sys.modules["matplotlib"].use = lambda *a, **kw: None
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
for name in ("roc_auc_score", "brier_score_loss", "log_loss"):
    setattr(sys.modules["sklearn.metrics"], name, lambda *a, **kw: None)
sys.modules["sklearn"].metrics = sys.modules["sklearn.metrics"]
prefix, order, root, names_json = sys.argv[1:]
names = json.loads(names_json)
before_path = list(sys.path)
before_cwd = pathlib.Path.cwd()
for name in names:
    old_name = prefix + name
    new_name = prefix + "reporting." + name
    first, second = (old_name, new_name) if order == "old-first" else (new_name, old_name)
    a = importlib.import_module(first)
    b = importlib.import_module(second)
    assert a is b
    assert sys.modules[old_name] is sys.modules[new_name]
    assert a.__name__ == new_name
    assert a.ROOT == pathlib.Path(root)
    assert a.OUT.is_relative_to(a.ROOT)
    # Verify function globals follow monkeypatches through either import spelling.
    original = a.ROOT
    token = object()
    b.ROOT = token
    assert a.main.__globals__["ROOT"] is token
    a.ROOT = original
assert sys.path == before_path
assert pathlib.Path.cwd() == before_cwd
print(json.dumps({"imports": len(names), "order": order, "prefix": prefix}))
'''


class ReportingLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).with_name("baseline_ast_sha256.json"))
                                 .read_text(encoding="utf-8"))
        cls.current = {name: ast.parse((MASTER / MAPPING[f"scripts/{name}.py"]).read_bytes())
                       for name in NAMES}

    def run_python(self, args, cwd):
        result = subprocess.run(
            [sys.executable, "-B", *args], cwd=cwd,
            env={**os.environ, **THREAD_ENV}, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.splitlines()

    def test_manifest_is_exact_and_paths_resolve(self):
        layout = json.loads((MASTER / "module-layout.json").read_text(encoding="utf-8"))
        self.assertEqual(layout["schema_version"], 1)
        self.assertEqual(layout["revision"], "master-reporting-layout-20261024")
        self.assertEqual(layout["baseline"], BASELINE)
        self.assertEqual(layout["group"], "reporting")
        self.assertEqual(layout["modules"], MAPPING)
        self.assertEqual(self.fixture["baseline"], BASELINE)
        self.assertEqual(set(self.fixture["modules"]), set(NAMES))
        for old, new in MAPPING.items():
            self.assertTrue((MASTER / old).is_file())
            self.assertTrue((MASTER / new).is_file())
            self.assertEqual((MASTER / old).resolve().parents[1],
                             (MASTER / new).resolve().parents[2])

    def test_complete_algorithm_asts_match_fixed_baseline(self):
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(self.fixture["modules"][name]["algorithm_ast_sha256"],
                                 fingerprint(normalized_canonical(self.current[name], name)))

    def test_ast_gate_rejects_changed_algorithm(self):
        for name in NAMES:
            with self.subTest(name=name):
                changed = copy.deepcopy(self.current[name])
                main = next(n for n in changed.body if isinstance(n, ast.FunctionDef)
                            and n.name == "main")
                main.body.insert(0, ast.parse("raise RuntimeError('algorithm changed')").body[0])
                self.assertNotEqual(self.fixture["modules"][name]["algorithm_ast_sha256"],
                                    fingerprint(normalized_canonical(changed, name)))

    def test_ast_gate_rejects_wrong_root_or_extra_provenance(self):
        wrong_root = copy.deepcopy(self.current[NAMES[0]])
        node = next(n for n in wrong_root.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "ROOT")
        node.value = ast.parse("Path(__file__).resolve().parents[3]", mode="eval").body
        with self.assertRaisesRegex(AssertionError, "ROOT"):
            normalized_canonical(wrong_root, NAMES[0])
        changed = copy.deepcopy(self.current["report_validation_suite"])
        addition = next(n for n in ast.walk(changed) if dump(n) == dump(added_inventory()))
        addition.value.elts.append(ast.Constant("unexpected.py"))
        with self.assertRaisesRegex(AssertionError, "provenance"):
            normalized_canonical(changed, "report_validation_suite")

    def test_provenance_adds_exact_sources_and_retains_previous_inventory(self):
        # Execute only the two inventory statements against a synthetic scripts
        # directory. No output/model/input path is opened or globbed.
        def initial_files(tree):
            return next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "files" for t in n.targets))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts/reporting").mkdir(parents=True)
            for path in MAPPING:
                (root / path).touch()
            (root / "scripts/run_validation_suite.py").touch()
            for path in MAPPING.values():
                (root / path).touch()
            before, after = {"ROOT": root}, {"ROOT": root}
            current = initial_files(self.current["report_validation_suite"])
            self.assertEqual(fingerprint(current), self.fixture["modules"]["report_validation_suite"]
                             ["legacy_inventory_ast_sha256"])
            actual_addition = next(n for n in ast.walk(self.current["report_validation_suite"])
                                   if dump(n) == dump(added_inventory()))
            exec(compile(ast.Module(body=[current], type_ignores=[]), "legacy-inventory", "exec"), before)
            exec(compile(ast.Module(body=[current, actual_addition], type_ignores=[]),
                         "canonical-inventory", "exec"), after)
            self.assertEqual(after["files"], before["files"] + [root / p for p in MAPPING.values()])
            for name in ("report_validation_suite", "report_window_validation"):
                self.assertIn(root / f"scripts/{name}.py", after["files"])
                self.assertIn(root / f"scripts/reporting/{name}.py", after["files"])

    def test_actual_import_identity_globals_root_and_main_guard(self):
        for prefix, cwd in (("master_ver.scripts.", REPO), ("scripts.", MASTER),
                            ("", MASTER / "scripts")):
            for order in ("old-first", "canonical-first"):
                with self.subTest(prefix=prefix, order=order):
                    output = self.run_python(["-c", IMPORT_CHECK, prefix, order,
                                              str(MASTER), json.dumps(NAMES)], cwd)
                    self.assertEqual(json.loads(output[0])["imports"], len(NAMES))

    def test_synthetic_cli_preserves_arguments_cwd_and_runpy_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            reporting = scripts / "reporting"
            reporting.mkdir(parents=True)
            cwd = root / "unrelated-working-directory"
            cwd.mkdir()
            fixture = ("import json, os, sys\n"
                       "if __name__ == '__main__':\n"
                       "    print(json.dumps(dict(name=__name__, file=__file__, argv=sys.argv, cwd=os.getcwd())))\n"
                       "else:\n"
                       "    raise AssertionError('CLI fixture imported without main guard')\n")
            for name in NAMES:
                with self.subTest(name=name):
                    old, new = scripts / (name + ".py"), reporting / (name + ".py")
                    old.write_bytes((MASTER / "scripts" / old.name).read_bytes())
                    new.write_text(fixture, encoding="utf-8")
                    args = ["--synthetic-only", "value with spaces", "--literal=한글", "--", "-x"]
                    payload = json.loads(self.run_python([str(old), *args], cwd)[0])
                    self.assertEqual(payload, dict(name="__main__", file=str(new),
                                     argv=[str(new), *args], cwd=str(cwd)))
                    outer = ("import json, runpy, sys; before=list(sys.argv); "
                             "runpy.run_path(sys.argv[1], run_name='__main__'); "
                             "print(json.dumps(dict(restored=sys.argv == before)))")
                    lines = self.run_python(["-c", outer, str(old), *args], cwd)
                    self.assertTrue(json.loads(lines[1])["restored"])
                    self.assertEqual(json.loads(lines[0])["argv"][1:], [str(old), *args])

    def test_syntax_and_tiny_launchers(self):
        for name in NAMES:
            path = MASTER / "scripts" / (name + ".py")
            source = path.read_text(encoding="utf-8")
            self.assertLessEqual(len(source.splitlines()), 12)
            compile(source, str(path), "exec")
            compile(self.current[name], str(MASTER / MAPPING[f"scripts/{name}.py"]), "exec")
            self.assertNotIn("sys.path", source)
            self.assertNotIn("meta_path", source)


if __name__ == "__main__":
    unittest.main()
