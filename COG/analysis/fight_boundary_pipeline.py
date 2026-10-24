"""Compatibility alias for :mod:`analysis.fight_boundaries.fight_boundary_pipeline`."""

import sys as _sys
from .fight_boundaries import fight_boundary_pipeline as _implementation

_sys.modules[__name__] = _implementation
