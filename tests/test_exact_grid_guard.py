"""Tests for gameplay/grid_guard.py (v4-exact plan test 2: the 5-s grid raises outside comparison_gates)."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gameplay.fight_clustering as FC
import gameplay.fights as F
from gameplay import comparison_gates as CG
from gameplay import grid_guard as GG
from gameplay.exact_population import ExactParams, detect_engagements_exact
from gameplay.grid_guard import GridForbiddenError, allow_grid, forbid_grid, grid_forbidden
from gameplay.setup_features import build_setup
from gameplay.state_value_v3 import StateBuilderV3
from tests.test_exact_state_v3 import CACHE, _load, kill, synth

R2_LIST = Path("C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/records/baselines/matches_r2_1500.txt")
V4_MODULES = ("event_survival", "respawn_rules", "item_state", "champion_attributes", "objective_timers",
              "exact_population", "cohorts_exact", "role_inference", "labels_exact", "state_value_v3",
              "setup_features")
GRID_NAMES = {"build_5s_position_grid", "_build_5s_position_grid", "_build_5s_position_grid_impl", "allow_grid",
              "respawn_aware_grid", "comparison_gates", "grid_guard"}
XY = np.zeros((3, 10, 2), dtype=np.float32)
MT = np.array([0, 60_000, 120_000], dtype=np.int64)


def _call_all_grids():
    errs = 0
    for fn in (lambda: FC.build_5s_position_grid(XY, MT, [], {}),
               lambda: F._build_5s_position_grid(XY, MT, [], {}),
               lambda: F._build_5s_position_grid_impl(xy_minute=XY, minute_ts=MT, kill_events=[], tm={})):
        with pytest.raises(GridForbiddenError):
            fn()
        errs += 1
    return errs


def test_forbid_grid_makes_every_grid_path_raise_and_restores():
    originals = [getattr(m, a) for m, a in GG.GRID_TARGETS]
    assert not grid_forbidden()
    FC.build_5s_position_grid(XY, MT, [], {})                  # callable outside the guard
    with forbid_grid():
        assert grid_forbidden()
        assert _call_all_grids() == 3
    assert [getattr(m, a) for m, a in GG.GRID_TARGETS] == originals
    with pytest.raises(ZeroDivisionError):                     # restored on error too
        with forbid_grid():
            1 / 0
    assert [getattr(m, a) for m, a in GG.GRID_TARGETS] == originals and not grid_forbidden()


def test_nested_forbid_restores_outer_state():
    with forbid_grid():
        with forbid_grid():
            assert grid_forbidden()
        assert grid_forbidden()
        _call_all_grids()
    assert not grid_forbidden()


def test_allow_grid_only_inside_comparison_gates():
    with pytest.raises(GridForbiddenError, match="comparison_gates"):
        allow_grid()
    with forbid_grid():
        with pytest.raises(GridForbiddenError):
            with allow_grid():
                pass
        _call_all_grids()
    assert GG.ALLOWED_CALLERS == frozenset({"gameplay.comparison_gates"})


def _synthetic_gate_pack():
    pack = synth([kill(250_000, 6, 1, assists=(2,), x=7000, y=7000)], n_frames=12)
    pack["xy_raw_minute"] = np.asarray(pack.get("xy_raw_minute", np.zeros((12, 10, 2))), dtype=np.float32) + 500.0
    pack["meta"] = dict(pack["meta"], patch="15.14", match_id="SYN_1")
    return pack


def test_v4_paths_run_under_forbid_grid_synthetic():
    pack = _synthetic_gate_pack()
    roles = {p: p - 1 for p in range(1, 11)}
    with forbid_grid():
        st = StateBuilderV3(pack, "15.14").at(300_000, setup=True)
        assert st.setup is not None and len(st.values) > 0
        s = build_setup(pack, 300_000, roles)
        assert len(s) > 0
        tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
        recs, diag = detect_engagements_exact(pack, tm, ExactParams(13700, 4264.0), "v4")
        assert isinstance(recs, list) and "finals" in diag
        # comparison_gates is the one allowed path, and the guard is back afterwards
        ctx = CG.gate_context(pack, tm, SimpleNamespace(TF2_VALIDITY_RADIUS=1600.0, COORD_NORM_DIV=16000.0))
        g = CG.gate_decisions(ctx, 300_000, (7000.0, 7000.0))
        assert set(g) == {"G1_pass", "G1_nb", "G1_nr", "G5_pass", "G5_nb", "G5_nr"}
        assert grid_forbidden()
        _call_all_grids()


@pytest.fixture(scope="module")
def real_pack():
    if not (CACHE.exists() and R2_LIST.exists()):
        pytest.skip("15.14 cache not available")
    mid = R2_LIST.read_text(encoding="utf-8").split()[0]
    pack = _load(mid, xy=True)                                  # asserts patch 15.14
    pack["meta"]["match_id"] = mid
    return pack


def test_v4_paths_run_under_forbid_grid_real(real_pack):
    pack = real_pack
    tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
    with forbid_grid():
        recs, diag = detect_engagements_exact(pack, tm, ExactParams(13700, 4264.0), "v4")
        assert recs and diag["finals"] == len(recs)
        b = StateBuilderV3(pack, "15.14")
        roles = {p: p - 1 for p in range(1, 11)}
        for rec in recs[:5]:
            st = b.at(int(rec["tau"]) - 1, setup=True)
            assert st.setup is not None
            assert len(build_setup(pack, int(rec["tau"]) - 1, roles)) > 0
        ctx = CG.gate_context(pack, tm)
        rows = CG.gates_for_records(ctx, recs)
        assert len(rows) == len(recs)
        _call_all_grids()
        with pytest.raises(GridForbiddenError):             # the v3.3 detector builds the grid
            F.detect_fights(dict(pack), tm)
    assert not grid_forbidden()


def test_v4_modules_never_name_the_grid():
    """AST check (docstrings and comments are not code): no v4 module refers to the grid or to the
    comparison gates."""
    root = Path(F.__file__).resolve().parent
    for mod in V4_MODULES:
        tree = ast.parse((root / f"{mod}.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names |= {a.name.split(".")[-1] for a in node.names}
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module.split(".")[-1])
        assert not (names & GRID_NAMES), (mod, names & GRID_NAMES)
