"""Compatibility alias for :mod:`analysis.fight_boundaries.map_regions`."""

import sys as _sys
from .fight_boundaries import map_regions as _implementation

_sys.modules[__name__] = _implementation
