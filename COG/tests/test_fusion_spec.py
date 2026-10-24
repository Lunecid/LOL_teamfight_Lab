"""Pinned-source parser equivalence without importing runner or its heavy dependencies."""
import argparse
import ast
from copy import deepcopy
from contextlib import redirect_stderr
import importlib.util
import io
import itertools
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[2]
BASE = "282dcc36f2e175b33782fc0b4bcb31581e663612"
FUNCTIONS = {"_parse_model_list", "_stable_unique", "_axis_key_from_token", "_normalize_fusion_axis_key",
             "_expand_fusion_axis_values", "_expand_single_fusion_expr", "_expand_fusion_exprs"}


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
    return ((isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id.startswith("_FUSION_")))


def execute(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, "<pure-fusion-source>", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def sources():
    text = subprocess.check_output(["git", "-C", str(ROOT), "show", BASE + ":COG/runner.py"], encoding="utf-8")
    baseline = ast.parse(text)
    current = ast.parse((ROOT / "COG/runner.py").read_text(encoding="utf-8"))
    path = ROOT / "COG/app/fusion_spec.py"
    spec = importlib.util.spec_from_file_location("pure_fusion_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = execute([n for n in baseline.body if selected(n)],
                        {"itertools": itertools, "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple})
    return baseline, current, ast.parse(path.read_text(encoding="utf-8")), namespace, module


def outcome(call, *args):
    try:
        return "result", call(*args)
    except Exception as error:
        return "error", type(error), str(error)


def expected_runner_with_failure_exit(nodes):
    """Permit exactly the authorized exit-after-summary statement, retaining the whole function."""
    nodes = deepcopy(nodes)
    isolated = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "_run_isolated_per_model")
    failed = isolated.body[-1]
    assert isinstance(failed, ast.If) and isinstance(failed.test, ast.Name) and failed.test.id == "failed"
    assert len(failed.body) == 1 and isinstance(failed.body[0], ast.Expr)
    failed.body.append(ast.parse("raise SystemExit(1)").body[0])
    return nodes


def test_extracted_ast_aliases_and_all_other_runner_code_are_unchanged(sources):
    check_docstring_filter()
    baseline, current, extracted, _, module = sources
    assert executable_ast([n for n in baseline.body if selected(n)]) == executable_ast([n for n in extracted.body if selected(n)])
    old = [n for n in baseline.body if not selected(n) and not (isinstance(n, ast.Import) and any(a.name == "itertools" for a in n.names))]
    aliases = [n for n in current.body if isinstance(n, ast.ImportFrom) and n.module == "app.fusion_spec"]
    assert len(aliases) == 1
    new = [n for n in current.body if n not in aliases]
    assert executable_ast(expected_runner_with_failure_exit(old)) == executable_ast(new)
    expected = {n.name if isinstance(n, ast.FunctionDef) else n.target.id for n in baseline.body if selected(n)}
    assert {a.name for a in aliases[0].names} == expected
    assert all(a.asname in (None, a.name) and hasattr(module, a.name) for a in aliases[0].names)
    assert all(n.module in {"__future__", "typing"} for n in extracted.body if isinstance(n, ast.ImportFrom))
    assert all(a.name == "itertools" for n in extracted.body if isinstance(n, ast.Import) for a in n.names)


@pytest.mark.parametrize("value", [None, "", ";, ", " A ; B,,A ", [None, " a ", 2, ""], ("a", "b"), 7, {"a": 1, "b": 2}])
def test_model_list_dependency_is_identical(sources, value):
    _, _, _, baseline, module = sources
    assert outcome(module._parse_model_list, value) == outcome(baseline["_parse_model_list"], value)


def test_aliases_prefixes_and_axes_match_pinned_source(sources):
    _, _, _, baseline, module = sources
    tokens = [None, "", "unknown", " A:B=C ", "RNN_ALL", "global_all", "graph_all", "event_all", "lgbm_all"]
    for axis, name, prefixes in [("rnn", "GLOBAL", ("rnn_", "global_")), ("gnn", "GNN", ("gnn_",)),
                                  ("event", "EVENT", ("event_", "attn_")), ("logit", "LOGIT", ("logit_", "lgbm_"))]:
        values = list(baseline["_FUSION_" + name + "_ALIASES"])
        for value in [None, "", "all", "unknown", *values, *(p + v for p in prefixes for v in values)]:
            assert outcome(module._expand_fusion_axis_values, axis, value) == outcome(baseline["_expand_fusion_axis_values"], axis, value)
        tokens.extend(values)
        tokens.extend(p + v for p in prefixes for v in values)
    for token in tokens:
        assert outcome(module._axis_key_from_token, token) == outcome(baseline["_axis_key_from_token"], token)
    for token in [None, "", "rnn", "GLOBAL", "g", "gnn", "graph", "n", "event", "attn", "e", "logit", "lgbm", "use_logit", "use_lgbm_logit", "l", "bad"]:
        assert module._normalize_fusion_axis_key(token) == baseline["_normalize_fusion_axis_key"](token)
    assert outcome(module._expand_fusion_axis_values, "bad", "all") == outcome(baseline["_expand_fusion_axis_values"], "bad", "all")


@pytest.mark.parametrize("expr", [None, "", " + + ", "gru+sage+avg+on", " global:bigru + graph:gat + event:event_xattn + lgbm:no ",
                                  "rnn_all+graph_all+event_all+logit_all", "rnn=gru+rnn=bigru+rnn=gru+logit=1+logit=0",
                                  "unknown", "bad=all", "rnn=bad", "gnn=bad", "event=bad", "logit=bad", "rnn=", "a:b=c"])
def test_single_expression_results_order_and_errors_match(sources, expr):
    _, _, _, baseline, module = sources
    assert outcome(module._expand_single_fusion_expr, expr) == outcome(baseline["_expand_single_fusion_expr"], expr)


@pytest.mark.parametrize("raw", [None, "", "gru+sage;bigru+gat,gru+sage", ["gru+mean", None, "bigru+xattn"], ("rnn_all+logit_all", "logit=1"), "rnn=unknown"])
def test_multiple_expressions_and_deduplication_match(sources, raw):
    _, _, _, baseline, module = sources
    assert outcome(module._expand_fusion_exprs, raw) == outcome(baseline["_expand_fusion_exprs"], raw)
    assert module._stable_unique(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_cartesian_order_and_canonical_strings(sources):
    module = sources[-1]
    assert module._expand_single_fusion_expr("rnn=gru+rnn=bigru+logit_all") == [
        "layered_fusion@global=rnn_ugru+logit=1", "layered_fusion@global=rnn_ugru+logit=0",
        "layered_fusion@global=rnn_bigru+logit=1", "layered_fusion@global=rnn_bigru+logit=0"]
    assert module._expand_single_fusion_expr("event=xattn") == ["layered_fusion@event=event_xattn"]


def cli_namespace(tree, fusion):
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in {"build_argparser", "_apply_paper_preset"})
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("_PAPER_CORE4") for t in n.targets))]
    return execute(nodes, {**fusion, "argparse": argparse, "cfg": SimpleNamespace(MODE="all", MAX_MATCHES=None, SEEDS=[7])})


@pytest.mark.parametrize("argv", [[], ["--fusion", "gru+gnn_all"], ["--paper_preset", "core4_1seed", "--seed", "42"],
                                  ["--paper_preset", "core4_1seed_fast"], ["--paper_preset", "core4_optimal"],
                                  ["--paper_preset", "core4_1seed_fast", "--paper_max_matches", "321", "--max_matches", "9999"],
                                  ["--split_mode", "holdout_patch"], ["--paper_preset", "bad"], ["--not-an-option"]])
def test_cli_help_defaults_presets_and_errors_without_runner_import(sources, argv):
    baseline, current, _, old_fusion, module = sources
    old, new = cli_namespace(baseline, old_fusion), cli_namespace(current, vars(module))
    a, b = old["build_argparser"](), new["build_argparser"]()
    assert a.format_help() == b.format_help()

    def parse(parser, namespace):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            try:
                args = parser.parse_args(argv)
                namespace["_apply_paper_preset"](args)
                return vars(args), stderr.getvalue()
            except SystemExit as error:
                return error.code, stderr.getvalue()
    assert parse(a, old) == parse(b, new)


@pytest.mark.parametrize("parallel", [1, 2], ids=["serial", "parallel"])
@pytest.mark.parametrize("returncodes", [(0, 0, 0, 0, 0), (3, 0, -9, 0, 0)], ids=["success", "failure"])
def test_isolated_children_all_finish_and_failure_exits_after_full_summary(sources, capsys, parallel, returncodes):
    current = sources[1]
    models = ["model_a", "model_b", "model_c", "model_d", "model_e"]
    codes = dict(zip(models, returncodes))
    commands, events = [], []

    def remember(command):
        commands.append(command)
        assert command[:3] == ["python-fixture", "-u", "runner.py"]
        assert command[3:] == ["--seed", "7", "--models", command[-1]]
        return command[-1]

    def run(command):
        name = remember(command)
        events.append(("run", name))
        return SimpleNamespace(returncode=codes[name])

    class Process:
        def __init__(self, command, stdout, stderr):
            assert stdout is None and stderr is None
            self.name, self.returncode = remember(command), None
            events.append(("start", self.name))

        def wait(self):
            events.append(("wait", self.name))
            self.returncode = codes[self.name]
            return self.returncode

    namespace = execute([n for n in current.body if isinstance(n, ast.FunctionDef)
                         and n.name in {"_strip_isolate_args", "_run_isolated_per_model"}],
                        {"sys": SimpleNamespace(executable="python-fixture", argv=["runner.py"]),
                         "subprocess": SimpleNamespace(run=run, Popen=Process)})
    call = namespace["_run_isolated_per_model"]
    argv = ["--isolate", "--parallel", str(parallel), "--models", "ignored", "--seed", "7"]
    if any(returncodes):
        with pytest.raises(SystemExit) as failure:
            call(argv, models, parallel)
        assert failure.value.code == 1
    else:
        assert call(argv, models, parallel) is None
    assert [command[-1] for command in commands] == models
    if parallel == 1:
        assert events == [("run", name) for name in models]
    else:
        expected = []
        for start in range(0, len(models), parallel):
            batch = models[start:start + parallel]
            expected.extend(("start", name) for name in batch)
            expected.extend(("wait", name) for name in batch)
        assert events == expected
    summary = capsys.readouterr().out.split("[ISOLATE] Summary:\n", 1)[1].splitlines()
    statuses = ["SUCCESS" if code == 0 else f"FAILED (exit={code})" for code in returncodes]
    assert summary[:len(models)] == [f"  {name}: {status}" for name, status in zip(models, statuses)]
    failed = [name for name, code in zip(models, returncodes) if code]
    if failed:
        assert summary[-1] == f"[ISOLATE] {len(failed)} model(s) FAILED: {failed}"
    else:
        assert summary[-1] == f"[ISOLATE] All {len(models)} model(s) completed successfully."
