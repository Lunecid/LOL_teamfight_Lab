"""Compatibility alias for :mod:`analysis.fight_boundaries.kill_pairs`."""

import sys as _sys
from .fight_boundaries import kill_pairs as _implementation

_sys.modules[__name__] = _implementation
