"""Compatibility entry point; implementation lives in reporting."""
from importlib import import_module
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("reporting") / "report_event_boundary_cif.py"), run_name="__main__")
else:
    _prefix = f"{__package__}." if __package__ else ""
    sys.modules[__name__] = import_module(f"{_prefix}reporting.report_event_boundary_cif")
