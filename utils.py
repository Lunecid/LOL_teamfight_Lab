from importlib import import_module as _import_module

_module = _import_module("core.utils")
for _k, _v in _module.__dict__.items():
    if _k in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__", "__builtins__"}:
        continue
    globals()[_k] = _v

del _k, _v, _module, _import_module
