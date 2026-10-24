"""Compatibility alias for :mod:`analysis.fight_boundaries.boundary_spec`."""

import sys as _sys
from .fight_boundaries import boundary_spec as _implementation

_sys.modules[__name__] = _implementation
