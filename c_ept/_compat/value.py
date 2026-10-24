"""Compatibility alias for :mod:`c_ept.valuation.value`."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("c_ept.valuation.value")

_sys.modules[__name__] = _module
