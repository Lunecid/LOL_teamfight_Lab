"""Compatibility alias for :mod:`c_ept.checks.gate_s3`."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("c_ept.checks.gate_s3")

_sys.modules[__name__] = _module
