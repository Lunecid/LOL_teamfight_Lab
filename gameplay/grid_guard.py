"""Guard for the legacy 5-s position grid (v4-exact plan, test 2: "the 5-s grid must raise when it
is called outside comparison_gates").

forbid_grid()  context manager: replaces every module attribute through which the grid is reached
               (gameplay.fight_clustering.build_5s_position_grid, gameplay.fights._build_5s_position_grid
               and gameplay.fights._build_5s_position_grid_impl, the alias fights.py imports) with a stub
               that raises GridForbiddenError.  Restores the previous attributes on exit (also on error).
               Re-entrant: nested forbid_grid() blocks restore their own entry state.
allow_grid()   context manager for gameplay.comparison_gates ONLY (the caller's module is checked; any
               other caller gets GridForbiddenError).  Inside it the real grid functions are installed;
               on exit the attributes that were there before (stubs, if under forbid_grid) come back.

Scope.  The guard works on module attributes, so code that bound the function to a local name with
`from gameplay.fight_clustering import build_5s_position_grid` BEFORE forbid_grid() was entered is not
covered (scripts/run_killless_encounters.py imports it inside a function, i.e. at call time, and is
covered).  The v4 modules never import the grid at all (tests/test_exact_grid_guard.py greps them).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Tuple

from gameplay import fight_clustering as _FC
from gameplay import fights as _F

__all__ = ["GridForbiddenError", "forbid_grid", "allow_grid", "grid_forbidden", "GRID_TARGETS"]

ALLOWED_CALLERS = frozenset({"gameplay.comparison_gates"})

# (module, attribute) pairs through which the 5-s grid can be reached
GRID_TARGETS: Tuple[Tuple[object, str], ...] = (
    (_FC, "build_5s_position_grid"),
    (_F, "_build_5s_position_grid"),
    (_F, "_build_5s_position_grid_impl"),
)

_REAL: Dict[Tuple[int, str], Callable] = {(id(m), a): getattr(m, a) for m, a in GRID_TARGETS}


class GridForbiddenError(RuntimeError):
    """Raised when the 5-s position grid is called outside gameplay.comparison_gates."""


def _stub(name: str) -> Callable:
    def forbidden(*_a, **_k):
        raise GridForbiddenError(
            f"5-s position grid ({name}) is forbidden here; only gameplay.comparison_gates may build it")
    forbidden.__grid_forbidden__ = True  # type: ignore[attr-defined]
    forbidden.__name__ = f"forbidden_{name}"
    return forbidden


def _snapshot() -> List[Tuple[object, str, Callable]]:
    return [(m, a, getattr(m, a)) for m, a in GRID_TARGETS]


def _restore(saved: List[Tuple[object, str, Callable]]) -> None:
    for m, a, fn in saved:
        setattr(m, a, fn)


def grid_forbidden() -> bool:
    """True while any grid attribute is a forbid_grid stub."""
    return any(getattr(getattr(m, a), "__grid_forbidden__", False) for m, a in GRID_TARGETS)


@contextmanager
def forbid_grid() -> Iterator[None]:
    """Every call of the 5-s grid inside the block raises GridForbiddenError."""
    saved = _snapshot()
    for m, a in GRID_TARGETS:
        setattr(m, a, _stub(f"{getattr(m, '__name__', '?')}.{a}"))
    try:
        yield
    finally:
        _restore(saved)


def _check_caller(depth: int) -> None:
    caller = sys._getframe(depth).f_globals.get("__name__", "")
    if caller not in ALLOWED_CALLERS:
        raise GridForbiddenError(f"allow_grid() may only be used in gameplay.comparison_gates (called from {caller!r})")


class allow_grid:  # noqa: N801  (used as a context manager, like forbid_grid)
    """Real grid functions inside the block; only gameplay.comparison_gates may enter it."""

    def __init__(self) -> None:
        _check_caller(2)
        self._saved: List[Tuple[object, str, Callable]] = []

    def __enter__(self) -> "allow_grid":
        self._saved = _snapshot()
        for m, a in GRID_TARGETS:
            setattr(m, a, _REAL[(id(m), a)])
        return self

    def __exit__(self, *exc) -> bool:
        _restore(self._saved)
        return False

