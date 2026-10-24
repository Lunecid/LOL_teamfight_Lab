"""Compatibility alias for :mod:`c_ept.models.baselines_b0`."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("c_ept.models.baselines_b0")

_sys.modules[__name__] = _module
