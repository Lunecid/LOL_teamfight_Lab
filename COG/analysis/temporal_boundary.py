"""Compatibility alias for :mod:`analysis.fight_boundaries.temporal_boundary`."""

import sys as _sys
from .fight_boundaries import temporal_boundary as _implementation

_sys.modules[__name__] = _implementation
