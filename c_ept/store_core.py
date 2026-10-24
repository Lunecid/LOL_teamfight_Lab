"""Compatibility alias for :mod:`c_ept.data.store_core`."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("c_ept.data.store_core")

_sys.modules[__name__] = _module
