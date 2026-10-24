"""Layout/serialization contracts only; no research data or experiment execution."""

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest


COG = Path(__file__).resolve().parents[1]
PACKAGE = COG / "analysis" / "fight_boundaries"
# Baseline d9d7ac8 source ASTs, normalizing only the six sibling imports.
BASELINE_ASTS = {'ability_range': 'c6a3149ce035a9e787d001a3db11faafdbbc3ca8478f619544075f2bf5f7d97d',
 'boundary_spec': '7783cd580ce7e310528b4f0ad00b106ad7abe4ed7c6ece547ceffe27bd65edfc',
 'kill_pairs': '08848b87347b4977aae2377bc066f73493bfb4edb3760bd69b854be36fa07687',
 'map_regions': '052bdbecf81a6189d99740e41afe5ef120e53a0a2dc6975c3afe20dcfaee864f',
 'spatial_boundary': 'a4e2abbdb455d521d3245f6df7f1908dcf504aefc1e67339a1251eae8d24b8e2',
 'temporal_boundary': '51a748681e2e634010a5c610acbdc9715439aade67302cd481881343243d3691',
 'fight_boundary_pipeline': '6b8700385bbdb4e5add325aa4066a80d146c70e237060bed6a8b984af3b295e4'}


IMPORT_CHECK = r"""
import importlib
import pickle
from pathlib import Path
import socket
import sys
import urllib.request


def deny_network(*args, **kwargs):
    raise AssertionError("Network access is forbidden in organization checks")


socket.create_connection = deny_network
socket.socket.connect = deny_network
urllib.request.urlopen = deny_network
paths_before = list(sys.path)
hooks_before = list(sys.meta_path)
names = ('ability_range', 'boundary_spec', 'kill_pairs', 'map_regions', 'spatial_boundary', 'temporal_boundary', 'fight_boundary_pipeline')
canonical_first = sys.argv[1] == "canonical"
mode = sys.argv[2]
if mode == "mixed-source-first":
    roots = ("analysis", "COG.analysis")
elif mode == "mixed-qualified-first":
    roots = ("COG.analysis", "analysis")
else:
    roots = (mode,)
expected_package = Path(sys.argv[3]).resolve() / "analysis" / "fight_boundaries"
modules = {}
probes = {}
checked_globals = 0
for leaf in names:
    for prefix in roots:
        legacy_name = prefix + "." + leaf
        canonical_name = prefix + ".fight_boundaries." + leaf
        first, second = ((canonical_name, legacy_name) if canonical_first
                         else (legacy_name, canonical_name))
        first_module = importlib.import_module(first)
        second_module = importlib.import_module(second)
        assert first_module is second_module, (prefix, leaf)
        modules[prefix, leaf] = first_module
        probes[prefix, leaf] = object()
        first_module._organization_probe = probes[prefix, leaf]

# Recheck saved references after both roots and all transitive imports have loaded.
for (prefix, leaf), module in modules.items():
    legacy_name = prefix + "." + leaf
    canonical_name = prefix + ".fight_boundaries." + leaf
    assert importlib.import_module(legacy_name) is module, legacy_name
    assert importlib.import_module(canonical_name) is module, canonical_name
    assert sys.modules[legacy_name] is sys.modules[canonical_name] is module
    assert getattr(importlib.import_module(prefix), leaf) is module
    assert getattr(importlib.import_module(prefix + ".fight_boundaries"), leaf) is module
    assert module._organization_probe is probes[prefix, leaf]
    assert Path(module.__file__).resolve() == expected_package / (leaf + ".py")
    assert module.__spec__.name == canonical_name
    for name, value in tuple(vars(module).items()):
        if getattr(value, "__module__", None) != legacy_name:
            continue
        checked_globals += 1
        if hasattr(value, "__globals__"):
            assert value.__globals__ is vars(module), (prefix, leaf, name)
        # Construct historical names in memory; never read an existing payload.
        historical = ("c" + legacy_name + "\n" + name + "\n.").encode("ascii")
        assert pickle.loads(historical) is value, (prefix, leaf, name)
        serialized = pickle.dumps(value, protocol=0)
        assert serialized.startswith(("c" + legacy_name + "\n").encode("ascii"))
        assert pickle.loads(serialized) is value

import numpy as np

for prefix in roots:
    BoundarySpec = modules[prefix, "boundary_spec"].BoundarySpec
    MatchRecord = modules[prefix, "kill_pairs"].MatchRecord
    ValleyEstimate = modules[prefix, "temporal_boundary"].ValleyEstimate
    objects = (
        BoundarySpec("synthetic", [], 0, 0, 0, 18.0, "default"),
        MatchRecord("synthetic", "synthetic", np.array([0.0, 1.0]), 2),
        ValleyEstimate(5.0, 60.0, 18.0, 0.2, 2, 0.08, True),
    )
    for value in objects:
        for protocol in (0, pickle.HIGHEST_PROTOCOL):
            loaded = pickle.loads(pickle.dumps(value, protocol=protocol))
            assert type(loaded) is type(value)
            assert type(loaded).__module__.startswith(prefix + ".")
            assert ".fight_boundaries." not in type(loaded).__module__
            for field, expected in vars(value).items():
                actual = getattr(loaded, field)
                if isinstance(expected, np.ndarray):
                    np.testing.assert_array_equal(actual, expected)
                elif isinstance(expected, float) and np.isnan(expected):
                    assert np.isnan(actual)
                else:
                    assert actual == expected, field

# Patch both roots simultaneously; each function must retain its own globals.
original_fetches = {}
fetch_probes = {}
try:
    for prefix in roots:
        module = modules[prefix, "ability_range"]
        original_fetches[prefix] = module.fetch_json
        fetch_probes[prefix] = object()
        module.fetch_json = fetch_probes[prefix]
    for prefix in roots:
        old = importlib.import_module(prefix + ".ability_range")
        new = importlib.import_module(prefix + ".fight_boundaries.ability_range")
        assert old is new is modules[prefix, "ability_range"]
        assert new.fetch_json is fetch_probes[prefix]
        assert new.load_champion_full.__globals__["fetch_json"] is fetch_probes[prefix]
finally:
    for prefix, fetch in original_fetches.items():
        modules[prefix, "ability_range"].fetch_json = fetch

if len(roots) == 2:
    for leaf in names:
        source = modules["analysis", leaf]
        qualified = modules["COG.analysis", leaf]
        assert source is not qualified, leaf
        assert source.__dict__ is not qualified.__dict__, leaf
        assert source._organization_probe is not qualified._organization_probe
    for leaf, symbol in (("boundary_spec", "BoundarySpec"),
                         ("kill_pairs", "MatchRecord"),
                         ("temporal_boundary", "ValleyEstimate")):
        assert getattr(modules["analysis", leaf], symbol) is not getattr(modules["COG.analysis", leaf], symbol)
else:
    absent = "analysis" if roots[0] == "COG.analysis" else "COG.analysis"
    assert absent not in sys.modules, absent
for module in modules.values():
    del module._organization_probe
assert sys.path == paths_before
assert len(sys.meta_path) == len(hooks_before)
assert all(a is b for a, b in zip(sys.meta_path, hooks_before))
assert "torch" not in sys.modules
print(f"{mode}, {sys.argv[1]} first: {len(modules)} modules; {checked_globals} historical globals; {3 * len(roots)} synthetic dataclasses")
"""


class FightBoundaryLayoutTests(unittest.TestCase):
    def test_algorithm_asts_preserved(self):
        for name, expected in BASELINE_ASTS.items():
            with self.subTest(module=name):
                tree = ast.parse((PACKAGE / (name + ".py")).read_text(encoding="utf-8"))
                registration_import, registration_call = tree.body[-2:]
                self.assertEqual(ast.dump(registration_import), ast.dump(ast.parse("from . import _register_legacy_module").body[0]))
                self.assertEqual(ast.dump(registration_call), ast.dump(ast.parse("_register_legacy_module(__name__)").body[0]))
                tree.body = tree.body[:-2]
                actual = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual, expected)

    def test_old_paths_are_only_aliases(self):
        for name in BASELINE_ASTS:
            with self.subTest(module=name):
                actual = ast.parse((COG / "analysis" / (name + ".py")).read_text(encoding="utf-8"))
                expected = ast.parse(
                    '"Compatibility alias"\n'
                    'import sys as _sys\n'
                    f'from .fight_boundaries import {name} as _implementation\n'
                    '_sys.modules[__name__] = _implementation\n'
                )
                # The descriptive module docstring is not part of the executable alias contract.
                self.assertEqual(ast.dump(ast.Module(body=actual.body[1:], type_ignores=[])),
                                 ast.dump(ast.Module(body=expected.body[1:], type_ignores=[])))

    def check_import_order(self, order):
        for mode in ("analysis", "COG.analysis", "mixed-source-first", "mixed-qualified-first"):
            with self.subTest(order=order, mode=mode):
                env = os.environ.copy()
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
                    env[name] = "1"
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                # Test-only search roots: production code never changes sys.path.
                env.pop("PYTHONPATH", None)
                if mode.startswith("mixed-"):
                    env["PYTHONPATH"] = str(COG)
                cwd = COG if mode == "analysis" else COG.parent
                result = subprocess.run(
                    [sys.executable, "-c", IMPORT_CHECK, order, mode, str(COG)],
                    cwd=cwd, env=env, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(result.stdout.strip())

    def test_legacy_first_identity_globals_and_pickle(self):
        self.check_import_order("legacy")

    def test_canonical_first_identity_globals_and_pickle(self):
        self.check_import_order("canonical")


if __name__ == "__main__":
    unittest.main(verbosity=2)
