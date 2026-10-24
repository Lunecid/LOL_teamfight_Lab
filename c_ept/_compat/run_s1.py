"""Compatibility alias for :mod:`c_ept.workflows.run_s1`."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("c_ept.workflows.run_s1")

if __name__ == "__main__":
    raise SystemExit(_module.main())

_sys.modules[__name__] = _module
