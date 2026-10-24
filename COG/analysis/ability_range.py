"""Compatibility alias for :mod:`analysis.fight_boundaries.ability_range`."""

import sys as _sys
from .fight_boundaries import ability_range as _implementation

_sys.modules[__name__] = _implementation
