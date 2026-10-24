"""Tiny-array metric equivalence to pinned source; never import the training runner."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
RELATIVE = "master_ver/studies/v_production_20261004/train.py"
BASE = "282dcc36f2e175b33782fc0b4bcb31581e663612"
FUNCTIONS = {"_sigmoid", "raw_logloss_from_logits", "raw_logloss_from_probability"}


def executable_ast(node):
    """Ignore only leading string docstrings; retain every executable AST node."""
    class StripDocstrings(ast.NodeTransformer):
        def generic_visit(self, current):
            current = super().generic_visit(current)
            if isinstance(current, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = current.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
                    current.body = body[1:]
            return current

    if isinstance(node, list):
        node = ast.Module(body=node, type_ignores=[])
    return ast.dump(StripDocstrings().visit(deepcopy(node)))


def check_docstring_filter():
    before = ast.parse('''"old module"
class C:
    "old class"
    @staticmethod
    def f(x: str = "runtime"):
        "old function"
        "retained expression"
        return x
async def g():
    "old async function"
    return "runtime"
''')
    documented = ast.parse(ast.unparse(before).replace("old ", "new "))
    assert executable_ast(before) == executable_ast(documented)
    assert executable_ast(before.body) == executable_ast(documented.body)
    changed_expression = ast.parse(ast.unparse(documented).replace("retained expression", "changed expression"))
    assert executable_ast(before) != executable_ast(changed_expression)
    changed_literal = ast.parse(ast.unparse(documented).replace("runtime", "changed runtime"))
    assert executable_ast(before) != executable_ast(changed_literal)
    assert ast.dump(before) != ast.dump(documented)  # comparison does not mutate either input


def selected(node):
    return isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS


@pytest.fixture(scope="module")
def sources():
    baseline = ast.parse(subprocess.check_output(["git", "-C", str(ROOT), "show", BASE + ":" + RELATIVE], encoding="utf-8"))
    current = ast.parse((ROOT / RELATIVE).read_text(encoding="utf-8"))
    path = (ROOT / RELATIVE).with_name("metrics.py")
    spec = importlib.util.spec_from_file_location("pure_v_metrics_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pure = ast.Module(body=[n for n in baseline.body if selected(n)], type_ignores=[])
    namespace = {"np": np, "Any": Any}
    exec(compile(pure, "<pinned-pure-v-metrics>", "exec"), namespace)
    return baseline, current, ast.parse(path.read_text(encoding="utf-8")), namespace, module


def outcome(call, *args):
    with np.errstate(all="ignore"):
        try:
            return "result", call(*args)
        except Exception as error:
            return "error", type(error), str(error)


def test_function_bodies_aliases_and_all_other_training_code_unchanged(sources):
    check_docstring_filter()
    baseline, current, extracted, _, module = sources
    assert executable_ast([n for n in baseline.body if selected(n)]) == executable_ast([n for n in extracted.body if selected(n)])
    aliases = [n for n in current.body if isinstance(n, ast.ImportFrom) and n.module == "studies.v_production_20261004.metrics"]
    assert len(aliases) == 1
    assert {a.name for a in aliases[0].names} == FUNCTIONS
    assert all(a.asname in (None, a.name) and callable(getattr(module, a.name)) for a in aliases[0].names)
    assert executable_ast([n for n in baseline.body if not selected(n)]) == executable_ast([n for n in current.body if n not in aliases])
    assert not any(isinstance(n, ast.ClassDef) for n in extracted.body)
    old_calibrator = next(n for n in baseline.body if isinstance(n, ast.ClassDef) and n.name == "PositiveSlopeCalibrator")
    new_calibrator = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == "PositiveSlopeCalibrator")
    assert executable_ast(old_calibrator) == executable_ast(new_calibrator)  # definition and pickle module stay in train.py
    imports = [a.name for n in extracted.body if isinstance(n, ast.Import) for a in n.names]
    assert imports == ["numpy"]
    assert all(n.module in {"__future__", "typing"} for n in extracted.body if isinstance(n, ast.ImportFrom))


@pytest.mark.parametrize("values", [0., [], [-1000., -745., -1., -0., 0., 1., 745., 1000.],
                                    [float("-inf"), float("inf"), float("nan")],
                                    np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]])
def test_sigmoid_preserves_float64_shape_bits_and_extreme_behavior(sources, values):
    _, _, _, baseline, module = sources
    with np.errstate(all="ignore"):
        expected, actual = baseline["_sigmoid"](values), module._sigmoid(values)
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("logits,target", [([0., -2., 2.], [0, 1, 1]), ([-1000., 1000.], [0, 1]),
                                          (np.arange(12).reshape(3, 4)[:, ::2], [0, 1, 0, 1, 0, 1]),
                                          ([[0.], [1.]], [False, True]), ([], []), ([0.], []),
                                          ([float("nan")], [0]), ([float("inf")], [1]),
                                          ([0.], [0.5]), ([0.], [float("nan")])])
def test_logit_loss_and_validation_match_pinned_source(sources, logits, target):
    _, _, _, baseline, module = sources
    assert outcome(module.raw_logloss_from_logits, logits, target) == outcome(baseline["raw_logloss_from_logits"], logits, target)


@pytest.mark.parametrize("probability,target", [([0.2, 0.8], [0, 1]), ([0., 1., -1., 2.], [0, 1, 0, 1]),
                                               ([1e-300, 1. - 1e-15], [1, 0]),
                                               ([float("-inf"), float("inf")], [0, 1]),
                                               ([float("nan")], [0]), ([], []), ([0.5], []),
                                               ([0.5], [2]), ([[0.2], [0.8]], [0, 1])])
def test_probability_loss_clipping_and_errors_match_pinned_source(sources, probability, target):
    _, _, _, baseline, module = sources
    assert outcome(module.raw_logloss_from_probability, probability, target) == outcome(baseline["raw_logloss_from_probability"], probability, target)


def test_loss_remains_unit_row_mean_in_float64(sources):
    module = sources[-1]
    values = np.array([-2., 0., 3.], dtype=np.float64)
    labels = np.array([0., 1., 1.], dtype=np.float64)
    expected = float(np.mean(np.logaddexp(0.0, values) - labels * values))
    assert module.raw_logloss_from_logits(values, labels) == expected
    assert isinstance(module.raw_logloss_from_logits(values, labels), float)
