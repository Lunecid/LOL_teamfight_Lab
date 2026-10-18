"""Independent runtime profile and hardware preflight for the 2026-10-02 study."""

from .profiles import PROFILE_SCHEMA, PROFILE_VERSION, runtime_profile


def run_preflight(*args, **kwargs):
    # Avoid importing the child module before runpy executes its entry point.
    from .preflight import run_preflight as execute

    return execute(*args, **kwargs)

__all__ = ["PROFILE_SCHEMA", "PROFILE_VERSION", "run_preflight", "runtime_profile"]
