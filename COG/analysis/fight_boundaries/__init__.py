"""Fight-boundary geometry, estimation, specification, and orchestration."""

import sys as _sys
from types import FunctionType as _FunctionType


def _register_legacy_module(module_name):
    """Share module state and preserve historical function/class pickle roots."""
    module = _sys.modules[module_name]
    legacy_name = module_name.replace(".fight_boundaries.", ".", 1)
    parent_name, leaf = legacy_name.rsplit(".", 1)
    for value in tuple(vars(module).values()):
        if isinstance(value, (type, _FunctionType)) and value.__module__ == module_name:
            value.__module__ = legacy_name
    _sys.modules[legacy_name] = module
    setattr(_sys.modules[parent_name], leaf, module)
